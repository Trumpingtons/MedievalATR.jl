# A Kraken segmentation model, such as blla, Kraken's default, and its network computed as Kraken 7.2 computes it
# (kraken/lib/vgsl/spred.py): the page scaled to the model's height, the network's output upsampled to that size and
# turned into probabilities, one map for each kind of object the model finds.

"""
	Segmenter

A Kraken segmentation model: its layers, the height its pages are scaled to, its number of channels (3 for colour,
1 for grey), the map of its output that holds each kind of object (`class_mapping["aux"]` the starts and ends of the
lines, `["baselines"]` each kind of baseline, `["regions"]` each kind of region), whether its lines are toplines
(`nothing` for centre lines), the kinds of region that bound the outlines of the lines, its VGSL specification and
Kraken's metadata.
"""
struct Segmenter
	layers::Vector{Any}
	height::Int
	channels::Int
	class_mapping::Dict{String, Dict{String, Int}}
	topline::Union{Bool, Nothing}
	bounding_regions::Vector{String}
	vgsl::String
	metadata::Dict{String, Any}
end

function Base.show(io::IO, model::Segmenter)
	kinds = sum(length(model.class_mapping[k]) for k in ("baselines", "regions") if haskey(model.class_mapping, k); init = 0)
	print(io, "Segmenter(height ", model.height, ", ", kinds, " kinds of object, ", length(model.layers), " layers)")
end

"""
	load_segmenter(path) -> Segmenter

The Kraken segmentation model saved at `path` (a Core ML .mlmodel file, as Kraken writes it), such as blla.mlmodel,
Kraken's default. Its layers are read as `load_model` reads them, and its class mapping from Kraken's metadata.
"""
function load_segmenter(path::AbstractString)
	user, metadata, network = read_kraken_file(path)
	get(metadata, "model_type", nothing) == "segmentation" || error("$(path) is not a segmentation model.")
	height, width, channels = input_shape(user["vgsl"], path)
	height > 1 && width == 0 || error("$(path) does not scale its pages to a fixed height, which MedievalATR requires.")
	channels in (1, 3) || error("$(path) takes $(channels) channels; MedievalATR reads models of 1 or 3.")
	get(metadata, "one_channel_mode", nothing) == "1" && error("$(path) takes binarised pages, which MedievalATR does not make.")
	layers = read_layers(network, path)
	mapping = Dict{String, Dict{String, Int}}(String(kind) => Dict{String, Int}(String(name) => Int(index) for (name, index) in names)
		for (kind, names) in metadata["class_mapping"])
	bounding = get(metadata, "bounding_regions", nothing)
	Segmenter(layers, height, channels, mapping, get(metadata, "topline", false), bounding === nothing ? String[] : String.(bounding),
		user["vgsl"], metadata)
end

"""
	segmentation_input(model, image) -> (input, scaled)

The page `image` (bytes, bands × width × height) as Kraken's input transforms prepare it for `model`: in the model's
colour mode, resized by Pillow's Lanczos filter to the model's height and the width in proportion, then as the network
takes it, (width, height, channels, 1), from 0 to 1 with ink high (the largest value of the page less each value).
`scaled` is the resized page in grey (width, height), on which the outlines of the lines are computed.
"""
function segmentation_input(model::Segmenter, image::Array{UInt8, 3})
	bands, w, h = size(image)
	page = model.channels == 3 ? (bands == 3 ? image : repeat(image, 3, 1, 1)) : reshape(grey(image), 1, w, h)
	height = model.height
	# Kraken's _fixed_resize: int(w * oh / h), in double precision
	width = unsafe_trunc(Int, w * height / h)
	resized = Array{UInt8, 3}(undef, size(page, 1), width, height)
	Threads.@threads for b in 1:size(page, 1)
		resized[b, :, :] = resize(page[b, :, :], width, height)
	end
	x = Float32.(resized) .* Float32(1 / 255)
	x = maximum(x) .- x
	reshape(permutedims(x, (2, 3, 1)), width, height, size(page, 1), 1), grey(resized)
end

"""
	network_output(model, input; torch_threads=1) -> Array{Float32, 3}

The output of the network of `model` on `input` (as `segmentation_input` makes it): one map of logits (width, height)
for each class, at the network's own resolution, as PyTorch computes it with `torch_threads` threads.
"""
function network_output(model::Segmenter, input::Array{Float32, 4}; torch_threads::Int = 1)
	x = input
	for layer in model.layers
		x = layer isa GroupNorm ? layer(x; torch_threads) : layer(x)
	end
	x[:, :, :, 1]
end

"""
PyTorch's nearest index (`F.interpolate`, mode "nearest") of output `index` (from 0) when `insize` values are
upsampled to `outsize`, computed in single precision as PyTorch computes it.
"""
function nearest(index::Int, insize::Int, outsize::Int)
	outsize == insize && return index
	outsize == 2insize && return index >> 1
	scale = Float32(insize) / Float32(outsize)
	min(unsafe_trunc(Int, floor(Float32(index) * scale)), insize - 1)
end

"""
	heatmaps(output, width, height) -> Array{Float32, 3}

The probability maps of the classes (width, height, class), the network's `output` upsampled to `width` and `height`
by PyTorch's nearest interpolation, then put through a sigmoid, as Kraken computes its heatmaps.
"""
function heatmaps(output::Array{Float32, 3}, width::Int, height::Int)
	w, h, classes = size(output)
	columns = [nearest(i, w, width) + 1 for i in 0:width - 1]
	rows = [nearest(j, h, height) + 1 for j in 0:height - 1]
	maps = Array{Float32, 3}(undef, width, height, classes)
	Threads.@threads for k in 1:classes
		@inbounds for j in 1:height, i in 1:width
			maps[i, j, k] = NNlib.sigmoid(output[columns[i], rows[j], k])
		end
	end
	maps
end
