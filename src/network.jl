# The layers of a Kraken recognition network, computed as Kraken computes them with PyTorch. Arrays are in NNlib's
# order (width, height, channels, batch), which is PyTorch's (batch, channels, height, width) read backwards, so a
# PyTorch reshape or permute of Kraken's code is done here on the reversed dimensions.

"""
	Convolution

A two-dimensional convolution with its activation, padded by half the kernel as Kraken pads it, so that with a stride
of 1 it keeps the height and width. `weight` is (kernel width, kernel height, input channels, output channels), and
`stride` (width, height).
"""
struct Convolution
	weight::Array{Float32, 4}
	bias::Vector{Float32}
	pad::NTuple{2, Int}
	stride::NTuple{2, Int}
	activation::Symbol
end

function (layer::Convolution)(x::Array{Float32, 4})
	y = NNlib.conv(x, layer.weight; pad = layer.pad, stride = layer.stride, flipped = true) .+ reshape(layer.bias, 1, 1, :, 1)
	activate(y, layer.activation)
end

"""
	GroupNorm

PyTorch's group normalisation (Kraken's VGSL `Gn`): the channels split into `groups` groups, each normalised by the
mean and variance of its values over the whole image, then scaled by `weight` and shifted by `bias`, channel by
channel, with PyTorch's epsilon of 1e-5. PyTorch computes the moments one way on a contiguous tensor and another on a
channels-last one, which its tensor is after an LSTM along the height (`channels_last`); the second way depends on the
number of PyTorch's threads.
"""
struct GroupNorm
	groups::Int
	weight::Vector{Float32}
	bias::Vector{Float32}
	channels_last::Bool
end

GroupNorm(groups::Int, weight::Vector{Float32}, bias::Vector{Float32}) = GroupNorm(groups, weight, bias, false)

# PyTorch's moments of a row of single-precision values (RowwiseMoments in ATen's moments_utils.h, 2.14, as built for
# ARM64): Welford's algorithm on 4 lanes, over chunks of 16 vectors merged in a cascade, then the values left over, one
# by one, then the 4 lanes, with the fused multiply-adds of NEON and of clang's contraction. The mean and the variance
# are in single precision, which is what matters on a near-constant channel, where the variance is small.
const Lanes = NTuple{4, Float32}
const ZERO_LANES = (0f0, 0f0, 0f0, 0f0)

"""The moments (count, mean, M2) of two sets of 4 lanes merged, as AddMomentsVec merges them."""
@inline function merge_moments(m0_add::Int, m1_add::Lanes, m2_add::Lanes, m0::Int, m1::Lanes, m2::Lanes)
	n = m0 + m0_add
	c = n == 0 ? 0f0 : Float32(m0_add) / Float32(n)
	scale = Float32(m0)
	delta = m1_add .- m1
	c_delta = c .* delta
	(n, m1 .+ c_delta, fma.(delta .* scale, c_delta, m2 .+ m2_add))
end

"""ceil(log2(x)), and 1 for x up to 2, as ATen's CeilLog2."""
ceil_log2(x::Int) = x <= 2 ? 1 : 64 - leading_zeros(UInt64(x - 1))

"""The mean and the (biased) variance of `x` as PyTorch's RowwiseMoments computes them on the CPU, in single precision."""
function torch_moments(x::AbstractVector{Float32})
	N = length(x)
	n = N ÷ 4
	chunks = cld(n, 16)
	depth = ceil_log2(chunks)
	m0s = zeros(Int, depth)
	m1s = fill(ZERO_LANES, depth)
	m2s = fill(ZERO_LANES, depth)
	@inbounds for i in 0:chunks - 1
		count = min(16, n - 16i)
		m1 = ZERO_LANES
		m2 = ZERO_LANES
		for j in 0:count - 1
			k = 64i + 4j
			v = (x[k + 1], x[k + 2], x[k + 3], x[k + 4])
			delta = v .- m1
			m1 = fma.(1f0 / Float32(j + 1), delta, m1)
			m2 = fma.(delta, v .- m1, m2)
		end
		m0s[1], m1s[1], m2s[1] = merge_moments(count, m1, m2, m0s[1], m1s[1], m2s[1])
		mask = i + 1
		level = 1
		while level < depth && iseven(mask)
			m0s[level + 1], m1s[level + 1], m2s[level + 1] = merge_moments(m0s[level], m1s[level], m2s[level],
				m0s[level + 1], m1s[level + 1], m2s[level + 1])
			m0s[level], m1s[level], m2s[level] = 0, ZERO_LANES, ZERO_LANES
			mask >>= 1
			level += 1
		end
	end
	@inbounds for level in 2:depth
		m0s[1], m1s[1], m2s[1] = merge_moments(m0s[level], m1s[level], m2s[level], m0s[1], m1s[1], m2s[1])
	end
	m0 = 0
	mean = 0f0
	m2 = 0f0
	@inbounds for i in 4n + 1:N
		v = x[i]
		delta = v - mean
		m0 += 1
		mean += delta / Float32(m0)
		m2 = fma(delta, v - mean, m2)
	end
	@inbounds for lane in 1:4
		total = m0 + n
		c = total == 0 ? 0f0 : Float32(n) / Float32(total)
		delta = m1s[1][lane] - mean
		mean = fma(c, delta, mean)
		m2 = m2 + fma(delta * delta * c, Float32(m0), m2s[1][lane])
		m0 = total
	end
	mean, m2 / Float32(N)
end

"""
	(layer::GroupNorm)(x; torch_threads=1)

`x` normalised as PyTorch 2.14 normalises it on the CPU with `torch_threads` threads, which matters only to a
channels-last tensor.
"""
function (layer::GroupNorm)(x::Array{Float32, 4}; torch_threads::Int = 1)
	layer.channels_last && return channels_last_groupnorm(layer, x, torch_threads)
	width, height, channels, batch = size(x)
	per = channels ÷ layer.groups
	y = similar(x)
	flat = vec(x)
	block = width * height * per
	Threads.@threads for (g, b) in collect(Iterators.product(1:layer.groups, 1:batch))
		# the group's values, contiguous, in the order of PyTorch's NCHW tensor
		first = ((b - 1) * layer.groups + g - 1) * block
		mean, variance = torch_moments(view(flat, first + 1:first + block))
		# PyTorch adds its epsilon, a double, in double precision
		rstd = Float32(1 / sqrt(Float64(max(variance, 0f0)) + 1e-5))
		for c in (g - 1) * per + 1:g * per
			scale = rstd * layer.weight[c]
			shift = fma(-scale, mean, layer.bias[c])
			@inbounds for j in 1:height, i in 1:width
				y[i, j, c, b] = fma(scale, x[i, j, c, b], shift)
			end
		end
	end
	y
end

"""
`x` normalised as PyTorch's GroupNormKernelImplChannelsLastInternal normalises a channels-last tensor of at least 1024
pixels: the pixels split into one run for each of `threads` threads, each summing every channel's values and squares in
single precision, the sums of the runs added in turn, and the variance taken as the mean of the squares less the square
of the mean. (PyTorch sums smaller maps in another order; blla's maps are always larger.)
"""
function channels_last_groupnorm(layer::GroupNorm, x::Array{Float32, 4}, threads::Int)
	width, height, channels, batch = size(x)
	pixels = width * height
	per = channels ÷ layer.groups
	total = batch * pixels
	runs = min(threads, total)
	run = cld(total, runs)
	sums = zeros(Float32, channels, batch, runs)
	squares = zeros(Float32, channels, batch, runs)
	flat = reshape(x, pixels, channels, batch)
	Threads.@threads for t in 1:runs
		@inbounds for i in (t - 1) * run:min(total, t * run) - 1
			b, m = divrem(i, pixels)
			for c in 1:channels
				v = flat[m + 1, c, b + 1]
				sums[c, b + 1, t] += v
				squares[c, b + 1, t] += v * v
			end
		end
	end
	s = 1f0 / Float32(per * pixels)
	y = similar(x)
	for b in 1:batch, g in 1:layer.groups
		mean = 0f0
		square = 0f0
		for c in (g - 1) * per + 1:g * per, t in 1:runs
			mean += sums[c, b, t]
			square += squares[c, b, t]
		end
		mean *= s
		variance = max(fma(square, s, -(mean * mean)), 0f0)
		rstd = Float32(1 / sqrt(Float64(variance) + 1e-5))
		for c in (g - 1) * per + 1:g * per
			scale = rstd * layer.weight[c]
			shift = fma(-scale, mean, layer.bias[c])
			@inbounds for j in 1:height, i in 1:width
				y[i, j, c, b] = x[i, j, c, b] * scale + shift
			end
		end
	end
	y
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

Kraken's bidirectional LSTM along the width (VGSL `Lbx`), or along the height when `vertical` (VGSL `Lby`, which
Kraken saves as a permutation of the height and width before the LSTM): each row (each column) of each image is a
sequence, and the output at each step is the forward hidden state followed by the backward one.
"""
struct BidirectionalLSTM
	forward::LSTMDirection
	backward::LSTMDirection
	vertical::Bool
end

BidirectionalLSTM(forward::LSTMDirection, backward::LSTMDirection) = BidirectionalLSTM(forward, backward, false)

function (layer::BidirectionalLSTM)(x::Array{Float32, 4})
	layer.vertical && return permutedims(horizontal_lstm(layer, permutedims(x, (2, 1, 3, 4))), (2, 1, 3, 4))
	horizontal_lstm(layer, x)
end

function horizontal_lstm(layer::BidirectionalLSTM, x::Array{Float32, 4})
	width, height, _, batch = size(x)
	n = size(layer.forward.recurrent, 2)
	y = Array{Float32, 4}(undef, width, height, 2n, batch)
	if height * batch == 1
		sequence = permutedims(x[:, 1, :, 1])
		# the two directions are independent: the backward one runs on another thread when Julia has one
		task = Threads.@spawn run(layer.backward, sequence; backward = true)
		y[:, 1, 1:n, 1] = permutedims(run(layer.forward, sequence))
		y[:, 1, n + 1:2n, 1] = permutedims(fetch(task))
		return y
	end
	# many sequences: each on a thread of its own
	Threads.@threads for (r, b) in collect(Iterators.product(1:height, 1:batch))
		# local: the name is also the single sequence's above, which the threads would otherwise share
		local row = permutedims(x[:, r, :, b])
		y[:, r, 1:n, b] = permutedims(run(layer.forward, row))
		y[:, r, n + 1:2n, b] = permutedims(run(layer.backward, row; backward = true))
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
