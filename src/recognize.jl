# Reading the lines of a page: the page image as bytes, and the text of each line as Kraken 7.2's `rpred` reads it.

"""
	read_image(path) -> Array{UInt8, 3}

The image at `path` as bytes (bands, width, height): one band for a grey image, three (red, green, blue) for any other,
whose alpha, if any, is dropped.
"""
read_image(path::AbstractString) = image_bytes(FileIO.load(path))

function image_bytes(img::AbstractMatrix{<:ColorTypes.Colorant})
	height, width = size(img)
	if eltype(img) <: Union{ColorTypes.AbstractGray, ColorTypes.TransparentGray}
		out = Array{UInt8}(undef, 1, width, height)
		for y in 1:height, x in 1:width
			out[1, x, y] = reinterpret(ColorTypes.gray(ColorTypes.Gray{FixedPointNumbers.N0f8}(img[y, x])))
		end
	else
		out = Array{UInt8}(undef, 3, width, height)
		for y in 1:height, x in 1:width
			c = ColorTypes.RGB{FixedPointNumbers.N0f8}(img[y, x])
			out[1, x, y] = reinterpret(ColorTypes.red(c))
			out[2, x, y] = reinterpret(ColorTypes.green(c))
			out[3, x, y] = reinterpret(ColorTypes.blue(c))
		end
	end
	out
end

"""
	Prediction

The text read on a line: the line's `id`, the `text` and its `characters`, empty when the line could not be read.
"""
struct Prediction
	id::String
	text::String
	characters::Vector{Character}
end

"""
The characters read on `line` of `image`, none where Kraken emits an empty record: a line without a boundary, one it
cannot cut (outside the image, or a baseline shorter than 5 pixels), and one whose image is empty or of a single shade.
"""
function read_line(model::Recognizer, image::Array{UInt8, 3}, line::TextLine)
	line.boundary === nothing && return Character[]
	cut = try
		cut_line(image, line.baseline, line.boundary)
	catch error
		error isa ArgumentError || rethrow()
		return Character[]
	end
	w, h = size(cut, 2), size(cut, 3)
	# Pillow cannot scale an image to a width of 0
	(w == 0 || h == 0 || trunc(Int, w * model.height / h) == 0) && return Character[]
	x = prepare_line(cut, model.height)
	minimum(x) == maximum(x) && return Character[]
	decode(model, probabilities(model, x))
end

"""
	recognize(model, image, lines) -> Vector{Prediction}
	recognize(model, page::Page) -> Vector{Prediction}
	recognize(model, path) -> Vector{Prediction}

The text of each of `lines` (text lines with their baselines and boundaries) of `image` (bands, width, height), read by
`model` as Kraken reads it: each line cut out along its baseline and boundary, prepared for the network, and decoded
greedily. A line Kraken cannot read gives an empty prediction. With a `page` or the `path` of its ALTO or PAGE file, the
lines of that page, on its image. The lines are read on Julia's threads.
"""
function recognize(model::Recognizer, image::Array{UInt8, 3}, lines::AbstractVector{TextLine})
	predictions = Vector{Prediction}(undef, length(lines))
	Threads.@threads :dynamic for i in eachindex(lines)
		characters = read_line(model, image, lines[i])
		predictions[i] = Prediction(lines[i].id, transcription(characters), characters)
	end
	predictions
end

recognize(model::Recognizer, page::Page) = recognize(model, read_image(page.image), page.lines)

recognize(model::Recognizer, path::AbstractString) = recognize(model, read_page(path))

"""
	write_predictions(path, page, predictions)

Writes the text read on each line of `page` to the JSON file at `path`: the page's `image`, `width` and `height`, and for
each line its `id`, `baseline` and `boundary` (lists of [x, y] points, the boundary `null` when the file gives none), the
`text` read and the `transcription` the page file gives (empty when none).
"""
function write_predictions(path::AbstractString, page::Page, predictions::AbstractVector{Prediction})
	length(predictions) == length(page.lines) || throw(ArgumentError("$(length(predictions)) predictions for $(length(page.lines)) lines"))
	points(v) = v === nothing ? nothing : [[x, y] for (x, y) in v]
	lines = [Dict("id" => l.id, "baseline" => points(l.baseline), "boundary" => points(l.boundary), "text" => p.text, "transcription" => l.text)
		for (l, p) in zip(page.lines, predictions)]
	write(path, JSON.json(Dict("image" => page.image, "width" => page.width, "height" => page.height, "lines" => lines)))
	path
end
