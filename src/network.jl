# The layers of a Kraken recognition network, computed as Kraken computes them with PyTorch. Arrays are in NNlib's
# order (width, height, channels, batch), which is PyTorch's (batch, channels, height, width) read backwards, so a
# PyTorch reshape or permute of Kraken's code is done here on the reversed dimensions.

"""
	Convolution

A two-dimensional convolution with its activation, padded so that it keeps the height and width (Kraken pads by half
the kernel). `weight` is (kernel width, kernel height, input channels, output channels).
"""
struct Convolution
	weight::Array{Float32, 4}
	bias::Vector{Float32}
	pad::NTuple{2, Int}
	activation::Symbol
end

function (layer::Convolution)(x::Array{Float32, 4})
	y = NNlib.conv(x, layer.weight; pad = layer.pad, flipped = true) .+ reshape(layer.bias, 1, 1, :, 1)
	activate(y, layer.activation)
end

"""
The activation `name` applied to `y`. A sigmoid is not applied: Kraken keeps the outputs of a convolution with a sigmoid
as logits.
"""
function activate(y, name::Symbol)
	name === :relu && return NNlib.relu.(y)
	name === :tanh && return tanh.(y)
	name in (:linear, :sigmoid) && return y
	error("The activation $(name) is not one MedievalATR computes.")
end

"""
	MaxPool

A max-pooling of `window` (width, height) with `stride`, without padding, the last rows and columns dropped when they
do not fill a window, as in PyTorch.
"""
struct MaxPool
	window::NTuple{2, Int}
	stride::NTuple{2, Int}
end

(layer::MaxPool)(x::Array{Float32, 4}) = NNlib.maxpool(x, layer.window; stride = layer.stride, pad = 0)

"""
	Reshape

Kraken's reshape layer (VGSL `S`): it splits dimension `source` (in PyTorch's order, from 0) in two of sizes `part_a`
and `part_b` (-1 for the rest), and moves them into dimensions `high` and `low`, which it then merges.
"""
struct Reshape
	source::Int
	part_a::Int
	part_b::Int
	high::Int
	low::Int
end

"""The size of `x` in PyTorch's order."""
torchsize(x) = reverse(size(x))

"""PyTorch's `reshape` to `shape` (in PyTorch's order, one dimension possibly -1), done on the reversed dimensions."""
function torchreshape(x, shape)
	known = prod(s for s in shape if s != -1; init = 1)
	full = map(s -> s == -1 ? length(x) ÷ known : s, shape)
	reshape(x, reverse(full)...)
end

"""PyTorch's `permute` by `perm` (counted from 0, in PyTorch's order), done on the reversed dimensions."""
function torchpermute(x, perm)
	n = length(perm)
	permutedims(x, [n - perm[n - j + 1] for j in 1:n])
end

function (layer::Reshape)(x::Array{Float32, 4})
	shape = torchsize(x)
	src = layer.source
	y = torchreshape(x, (shape[1:src]..., layer.part_a, layer.part_b, shape[src + 2:end]...))
	dest = layer.low
	if layer.high != src
		dest = layer.high
	else
		src += 1
	end
	perm = collect(0:ndims(y) - 1)
	step = dest > src ? 1 : -1
	for k in src:step:dest - step
		perm[k + 1], perm[k + step + 1] = perm[k + step + 1], perm[k + 1]
	end
	y = torchpermute(y, perm)
	shape = torchsize(y)
	torchreshape(y, (shape[1:dest]..., shape[dest + 1] * shape[dest + 2], shape[dest + 3:end]...))
end

"""
	LSTMDirection

One direction of an LSTM: `input` (4 × hidden, inputs) and `recurrent` (4 × hidden, hidden), the gates stacked in
PyTorch's order (input, forget, cell, output), and the sum of both biases.
"""
struct LSTMDirection
	input::Matrix{Float32}
	recurrent::Matrix{Float32}
	bias::Vector{Float32}
end

"""
The hidden states of `direction` (hidden, time) over the sequence `x` (inputs, time), read from the end when
`backward`.
"""
function run(direction::LSTMDirection, x::AbstractMatrix{Float32}; backward::Bool = false)
	n, steps = size(direction.recurrent, 2), size(x, 2)
	gates = direction.input * x .+ direction.bias
	h, c, z = zeros(Float32, n), zeros(Float32, n), zeros(Float32, 4n)
	out = Matrix{Float32}(undef, n, steps)
	recurrent = direction.recurrent
	for t in (backward ? (steps:-1:1) : (1:steps))
		# the recurrent product by columns, without BLAS, whose threads cost more than this small product
		fill!(z, 0.0f0)
		@inbounds for j in 1:n
			hj = h[j]
			@simd for i in 1:4n
				z[i] += recurrent[i, j] * hj
			end
		end
		@inbounds for k in 1:n
			i = NNlib.sigmoid(z[k] + gates[k, t])
			f = NNlib.sigmoid(z[n + k] + gates[n + k, t])
			g = tanh(z[2n + k] + gates[2n + k, t])
			o = NNlib.sigmoid(z[3n + k] + gates[3n + k, t])
			c[k] = f * c[k] + i * g
			h[k] = o * tanh(c[k])
		end
		out[:, t] = h
	end
	out
end

"""
	BidirectionalLSTM

Kraken's bidirectional LSTM along the width (VGSL `Lbx`): each row of each image is a sequence, and the output at
each step is the forward hidden state followed by the backward one.
"""
struct BidirectionalLSTM
	forward::LSTMDirection
	backward::LSTMDirection
end

function (layer::BidirectionalLSTM)(x::Array{Float32, 4})
	width, height, _, batch = size(x)
	n = size(layer.forward.recurrent, 2)
	y = Array{Float32, 4}(undef, width, height, 2n, batch)
	for b in 1:batch, r in 1:height
		sequence = permutedims(x[:, r, :, b])
		# the two directions are independent: the backward one runs on another thread when Julia has one
		backward = Threads.@spawn run(layer.backward, sequence; backward = true)
		y[:, r, 1:n, b] = permutedims(run(layer.forward, sequence))
		y[:, r, n + 1:2n, b] = permutedims(fetch(backward))
	end
	y
end

"""
	Linear

Kraken's output layer (VGSL `O`): a linear map of the channels at each position, `weight` (outputs, inputs).
"""
struct Linear
	weight::Matrix{Float32}
	bias::Vector{Float32}
end

function (layer::Linear)(x::Array{Float32, 4})
	width, height, channels, batch = size(x)
	y = layer.weight * reshape(permutedims(x, (3, 1, 2, 4)), channels, :) .+ layer.bias
	permutedims(reshape(y, :, width, height, batch), (2, 3, 1, 4))
end
