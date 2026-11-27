# A Kraken recognition model, read from the Core ML file (.mlmodel) in which Kraken saves it: the layers in the order of
# the file, with their weights laid out as Kraken's layers.py loads them, the codec and Kraken's metadata.

"""
	Recognizer

A Kraken recognition model: its layers, the text of each output label (`codec[label + 1]`, "" for the blank, label 0),
the height its line images are scaled to (0 for none), its number of channels, its VGSL specification and Kraken's
metadata (`seg_type`, `one_channel_mode`, `hyper_params`...).
"""
struct Recognizer
	layers::Vector{Any}
	codec::Vector{String}
	height::Int
	channels::Int
	vgsl::String
	metadata::Dict{String, Any}
end

function Base.show(io::IO, model::Recognizer)
	print(io, "Recognizer(", length(model.codec), " labels, height ", model.height, ", ", length(model.layers), " layers)")
end

# The numbers of the fields of Core ML's .proto files (Model.proto, NeuralNetwork.proto) that a Kraken model uses.
const DESCRIPTION, METADATA, USER_DEFINED, NEURAL_NETWORK, LAYERS = 2, 100, 100, 500, 1
const KINDS = Dict(100 => :convolution, 120 => :pooling, 130 => :activation, 140 => :inner_product, 175 => :softmax,
	310 => :permute, 430 => :bidirectional_lstm, 500 => :custom)
const ACTIVATIONS = Dict(5 => :linear, 10 => :relu, 30 => :tanh, 40 => :sigmoid)

"""The kind of the layer `layer` of a Core ML network and the message of its parameters."""
function layerkind(layer::Bytes)
	for f in fields(layer)
		haskey(KINDS, f.number) && return KINDS[f.number], f.bytes
	end
	error("The layer $(text(layer, 1)) is of a kind MedievalATR does not read.")
end

"""The weights of field `number` of the parameters `params` (a WeightParams message, its floats in field 1)."""
function weights(params::Bytes, number::Int)
	found = message(params, number)
	found === nothing ? Float32[] : floats(found, 1)
end

"""
A matrix of `rows` rows from `values`, stored row by row as PyTorch stores a weight (rows, columns).
"""
rowmajor(values::Vector{Float32}, rows::Int) = permutedims(reshape(values, :, rows))

"""One direction of a Core ML LSTM (an LSTMWeightParams message) as Kraken loads it into PyTorch."""
function lstmdirection(params::Bytes, hidden::Int)
	input = reduce(vcat, rowmajor(weights(params, n), hidden) for n in (1, 2, 3, 4))
	recurrent = reduce(vcat, rowmajor(weights(params, n), hidden) for n in (20, 21, 22, 23))
	bias = reduce(vcat, weights(params, n) for n in (40, 41, 42, 43))
	isempty(bias) && (bias = zeros(Float32, 4hidden))
	LSTMDirection(input, recurrent, bias)
end

"""The parameters of a Core ML custom layer (a map of names to CustomLayerParamValue messages) as integers and floats."""
function customparameters(params::Bytes)
	found = Dict{String, Any}()
	for entry in messages(params, 30)
		value = message(entry, 2)
		found[text(entry, 1)] = any(f.number == 10 for f in fields(value)) ? double(value, 10) : integer(value, 30)
	end
	found
end

"""
The layers of the Core ML network `network`, in the order of the file, with their weights laid out as Kraken's
layers.py loads them: convolutions (with their stride and activation), max-pooling, reshape, group normalisation,
bidirectional LSTMs along the width or, after a permutation of the height and width, along the height, and the output
layer. Another kind of layer is an error; `source` names the file in the error.
"""
function read_layers(network::Bytes, source::AbstractString)
	layers = Any[]
	vertical = false
	# whether PyTorch's tensor is channels-last at this point of Kraken's network: an LSTM along the height leaves it so,
	# a convolution, a group normalisation and a max-pooling keep it, other layers make it contiguous again
	channels_last = false
	for layer in messages(network, LAYERS)
		kind, params = layerkind(layer)
		name = text(layer, 1)
		if kind === :convolution
			integer(params, 60) == 0 || error("Layer $(name) of $(source) is a transposed convolution, which MedievalATR does not compute.")
			all(==(1), integers(params, 40)) ||
				error("Layer $(name) of $(source) is a dilated convolution, which MedievalATR does not compute.")
			out, in = integer(params, 1), integer(params, 2)
			kh, kw = integers(params, 20)
			strides = integers(params, 30)
			sh, sw = isempty(strides) ? (1, 1) : strides
			weight = reshape(weights(params, 90), kw, kh, in, out)
			bias = integer(params, 70) == 1 ? weights(params, 91) : zeros(Float32, out)
			push!(layers, Convolution(weight, bias, ((kw - 1) ÷ 2, (kh - 1) ÷ 2), (sw, sh), :linear))
		elseif kind === :activation
			# Kraken writes a convolution's activation as the layer after it
			activation = first(ACTIVATIONS[f.number] for f in fields(params) if haskey(ACTIVATIONS, f.number))
			previous = last(layers)
			previous isa Convolution || error("Layer $(name) of $(source) is an activation that does not follow a convolution.")
			layers[end] = Convolution(previous.weight, previous.bias, previous.pad, previous.stride, activation)
		elseif kind === :pooling
			integer(params, 1) == 0 || error("Layer $(name) of $(source) is a pooling other than max-pooling.")
			kh, kw = integers(params, 10)
			sh, sw = integers(params, 20)
			push!(layers, MaxPool((kw, kh), (sw, sh)))
		elseif kind === :permute
			# Kraken permutes the height and width before an LSTM along the height (VGSL `Lby`), and only there
			integers(params, 1) == [0, 1, 3, 2] || error("Layer $(name) of $(source) is a permutation other than Kraken's of the height and width.")
			vertical = true
		elseif kind === :custom
			class = text(params, 10)
			parameters = customparameters(params)
			if class == "reshape"
				push!(layers, Reshape(parameters["src_dim"], parameters["part_a"], parameters["part_b"], parameters["high"], parameters["low"]))
				channels_last = false
			elseif class == "groupnorm"
				scale, shift = (floats(w, 1) for w in messages(params, 20))
				push!(layers, GroupNorm(parameters["num_groups"], scale, shift, channels_last))
			elseif class != "dropout"
				error("Layer $(name) of $(source) is a custom layer $(class), which MedievalATR does not compute.")
			end
		elseif kind === :bidirectional_lstm
			hidden = integer(params, 2)
			directions = messages(params, 20)
			push!(layers, BidirectionalLSTM(lstmdirection(directions[1], hidden), lstmdirection(directions[2], hidden), vertical))
			channels_last = vertical
			vertical = false
		elseif kind === :inner_product
			out = integer(params, 2)
			bias = integer(params, 10) == 1 ? weights(params, 21) : zeros(Float32, out)
			push!(layers, Linear(rowmajor(weights(params, 20), out), bias))
			channels_last = false
		end
		# a softmax layer is not computed here: probabilities applies it, as Kraken's recognizer does
		vertical && kind !== :permute && error("Layer $(name) of $(source) follows a permutation but is not an LSTM.")
	end
	layers
end

"""
The VGSL specification, Kraken's metadata and the Core ML network of the Kraken model saved at `path`.
"""
function read_kraken_file(path::AbstractString)
	bytes = read(path)
	root = view(bytes, 1:length(bytes))
	description = message(root, DESCRIPTION)
	user = Dict(text(entry, 1) => text(entry, 2) for entry in messages(message(description, METADATA), USER_DEFINED))
	haskey(user, "vgsl") || error("$(path) has no VGSL specification: it is not a model Kraken saved.")
	network = message(root, NEURAL_NETWORK)
	network === nothing && error("$(path) holds no neural network.")
	metadata = haskey(user, "kraken_meta") ? JSON.parse(user["kraken_meta"]) : Dict{String, Any}()
	user, metadata, network
end

"""The height, width and channels of the input of the VGSL specification `vgsl` of the model at `path`."""
function input_shape(vgsl::AbstractString, path::AbstractString)
	shape = match(r"^\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", vgsl)
	shape === nothing && error("The VGSL specification of $(path) does not start with its input shape.")
	parse(Int, shape[2]), parse(Int, shape[3]), parse(Int, shape[4])
end

"""
	load_model(path) -> Recognizer

The Kraken recognition model saved at `path` (a Core ML .mlmodel file, as Kraken writes it, with its VGSL
specification, codec and metadata). Convolutions, max-pooling, reshape, bidirectional LSTMs along the width and the
output layer are read; another kind of layer is an error.
"""
function load_model(path::AbstractString)
	user, metadata, network = read_kraken_file(path)
	get(metadata, "model_type", "recognition") == "segmentation" &&
		error("$(path) is a segmentation model: load it with load_segmenter.")
	layers = read_layers(network, path)
	haskey(user, "codec") || error("$(path) has no codec: it is not a recognition model.")
	codec = JSON.parse(user["codec"])
	labels = fill("", maximum(maximum, values(codec)) + 1)
	for (code, label) in codec
		length(label) == 1 || error("The codec of $(path) maps a text to several labels, which MedievalATR does not decode.")
		labels[label[1] + 1] = code
	end
	height, _, channels = input_shape(user["vgsl"], path)
	Recognizer(layers, labels, height, channels, user["vgsl"], metadata)
end

"""
	probabilities(model, line) -> Matrix{Float32}

The probability of each label (rows, label 0 the blank first) at each step along the line (columns), for `line`, a
prepared line image (height, width) with values from 0 to 1, ink high, as Kraken's input transforms make it.
"""
function probabilities(model::Recognizer, line::AbstractMatrix{<:Real})
	x = reshape(Float32.(permutedims(line)), size(line, 2), size(line, 1), 1, 1)
	for layer in model.layers
		x = layer(x)
	end
	logits = permutedims(x[:, 1, :, 1])
	NNlib.softmax(logits; dims = 1)
end

"""
	Character

A character of a decoded line: its text, the first and last step (column) of the network output it was read from, and its
confidence (the highest probability of its label over those steps).
"""
struct Character
	text::String
	first::Int
	last::Int
	confidence::Float32
end

"""
	decode(model, probabilities) -> Vector{Character}

The characters of the line, by greedy (best path) CTC decoding as Kraken's `greedy_decoder`: the most probable label
at each step, runs of the same label merged, blanks dropped.
"""
function decode(model::Recognizer, probabilities::AbstractMatrix{Float32})
	characters = Character[]
	steps = size(probabilities, 2)
	t = 1
	while t <= steps
		best = argmax(view(probabilities, :, t))
		label, start, confidence = best - 1, t, probabilities[best, t]
		while t < steps && argmax(view(probabilities, :, t + 1)) == best
			t += 1
			confidence = max(confidence, probabilities[best, t])
		end
		label == 0 || push!(characters, Character(model.codec[label + 1], start, t, confidence))
		t += 1
	end
	characters
end

"""The text of the decoded `characters`."""
transcription(characters::Vector{Character}) = join(c.text for c in characters)
