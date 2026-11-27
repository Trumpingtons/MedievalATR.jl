# The lines of a page read from its ALTO or PAGE file, as Kraken 7.2's `XMLPage` reads them (kraken/lib/xml): the text
# lines of the regions, in the order of the file, each with its baseline, boundary polygon and transcription.

"""
	TextLine

A line of a page: its `id`, its `baseline` and `boundary` polygon as integer (x, y) points on the page (`nothing` when
the file gives no polygon), and its `text`, the transcription the file gives (empty when none).
"""
struct TextLine
	id::String
	baseline::Vector{NTuple{2, Int}}
	boundary::Union{Nothing, Vector{NTuple{2, Int}}}
	text::String
end

"""
	Page

A page read from an ALTO or PAGE file with [`read_page`](@ref): the path of its `image`, its `width` and `height` as the
file gives them, and its `lines`, in the order of the file.
"""
struct Page
	image::String
	width::Int
	height::Int
	lines::Vector{TextLine}
end

Base.show(io::IO, page::Page) = print(io, "Page(", repr(basename(page.image)), ", ", length(page.lines), " lines)")

"""The points with each run of equal points kept once, as `itertools.groupby` keeps them."""
function adjacent_unique(points::Vector{NTuple{2, Int}})
	out = NTuple{2, Int}[]
	for p in points
		(isempty(out) || out[end] != p) && push!(out, p)
	end
	out
end

pairs_of(values) = [(values[i], values[i + 1]) for i in 1:2:length(values) - 1]

const ALTO_NUMBER = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"

"""
The points of an ALTO `POINTS` or `BASELINE` attribute, in any of ALTO's forms (`x0,y0 x1,y1`, `x0 y0 x1 y1`,
`(x0, y0) (x1, y1)`), each number truncated to an integer, as Kraken's `parse_alto_pointstype` reads them.
"""
function alto_points(s::AbstractString)
	values = [trunc(Int, parse(Float64, m.match)) for m in eachmatch(ALTO_NUMBER, s)]
	isodd(length(values)) && throw(ArgumentError("Odd number of points in points sequence: $(s)"))
	adjacent_unique(pairs_of(values))
end

"""The points of a PAGE `points` attribute (`x0,y0 x1,y1`), as Kraken's `parse_page_coords` reads them."""
page_points(s::AbstractString) = adjacent_unique(pairs_of([parse(Int, c) for p in split(s, ' ') for c in split(p, ',')]))

# the XPath queries name elements by their local names, so they need no namespace prefixes
const NO_NAMESPACES = Pair{String, String}[]

"""The children of `node` whose name (without namespace) ends with one of `names`."""
children_named(node, names...) = [c for c in EzXML.eachelement(node) if any(n -> endswith(EzXML.nodename(c), n), names)]

"""The first child of `node` named `name` (without namespace), `nothing` when there is none."""
function first_child(node, name)
	for c in EzXML.eachelement(node)
		EzXML.nodename(c) == name && return c
	end
	nothing
end

"""The first element of the document below `node` named `name` (without namespace), `nothing` when there is none."""
first_descendant(node, name) = EzXML.findfirst(".//*[local-name()='$(name)']", node, NO_NAMESPACES)

attribute(node, name) = EzXML.haskey(node, name) ? node[name] : nothing

"""The text of an ALTO String or SP: its CONTENT, or a space when it has none."""
function content(node)
	c = attribute(node, "CONTENT")
	c === nothing || isempty(c) ? " " : c
end

"""The points of `s` read by `parse`, `nothing` when `s` is missing or cannot be read, as Kraken skips them."""
function points_or_nothing(parse::Function, s)
	s === nothing && return nothing
	try
		parse(s)
	catch error
		error isa ArgumentError || rethrow()
		nothing
	end
end

"""The image path of a page file, or the file of that name next to it when the path does not exist."""
function image_path(folder, name)
	path = joinpath(folder, name)
	isfile(path) && return path
	beside = joinpath(folder, basename(name))
	isfile(beside) ? beside : path
end

const ALTO_REGIONS = ("TextBlock", "Illustration", "GraphicalElement", "ComposedBlock")

const PAGE_REGIONS = ("TextRegion", "ImageRegion", "LineDrawingRegion", "GraphicRegion", "TableRegion", "ChartRegion",
	"MapRegion", "SeparatorRegion", "MathsRegion", "ChemRegion", "MusicRegion", "AdvertRegion", "NoiseRegion",
	"UnknownRegion", "CustomRegion")

"""The lines of an ALTO document, as Kraken's `parse_alto` reads them for baseline models."""
function read_alto(root, folder)
	unit = first_descendant(root, "MeasurementUnit")
	unit !== nothing && strip(EzXML.nodecontent(unit)) != "pixel" &&
		throw(ArgumentError("The measurement unit of the ALTO file is \"$(strip(EzXML.nodecontent(unit)))\", not \"pixel\"."))
	file = first_descendant(root, "fileName")
	(file === nothing || isempty(EzXML.nodecontent(file))) && throw(ArgumentError("No image file name in the ALTO file."))
	page = first_descendant(root, "Page")
	page === nothing && throw(ArgumentError("No Page element in the ALTO file."))
	lines = TextLine[]
	for layout in children_named(root, "Layout"), p in children_named(layout, "Page"), space in children_named(p, "PrintSpace")
		for region in children_named(space, ALTO_REGIONS...), line in children_named(region, "TextLine")
			baseline = points_or_nothing(alto_points, attribute(line, "BASELINE"))
			# a line without a baseline is left out
			baseline === nothing && continue
			shape = first_child(line, "Shape")
			polygon = shape === nothing ? nothing : first_child(shape, "Polygon")
			boundary = polygon === nothing ? nothing : points_or_nothing(alto_points, attribute(polygon, "POINTS"))
			# the CONTENT of each String, and a space for each SP (or String without content), in the order of the file
			text = join(content(s) for s in EzXML.findall(".//*[local-name()='String'] | .//*[local-name()='SP']", line, NO_NAMESPACES))
			push!(lines, TextLine(something(attribute(line, "ID"), ""), baseline, boundary, text))
		end
	end
	Page(image_path(folder, EzXML.nodecontent(file)), parse(Int, page["WIDTH"]), parse(Int, page["HEIGHT"]), lines)
end

"""The lines of a PAGE document, as Kraken's `parse_page` reads them for baseline models."""
function read_page_xml(root, folder)
	page = first_descendant(root, "Page")
	(page === nothing || attribute(page, "imageFilename") === nothing) && throw(ArgumentError("No image file name in the PAGE file."))
	lines = TextLine[]
	for region in children_named(page, PAGE_REGIONS...), line in children_named(region, "TextLine")
		base = first_child(line, "Baseline")
		baseline = base === nothing ? nothing : points_or_nothing(page_points, attribute(base, "points"))
		baseline === nothing && continue
		coords = first_child(line, "Coords")
		boundary = coords === nothing ? nothing : points_or_nothing(page_points, attribute(coords, "points"))
		# the line's own TextEquiv when it has one, otherwise every Unicode element below the line (its words')
		equiv = first_child(line, "TextEquiv")
		text = join(EzXML.nodecontent(u) for u in EzXML.findall(".//*[local-name()='Unicode']", something(equiv, line), NO_NAMESPACES))
		push!(lines, TextLine(something(attribute(line, "id"), ""), baseline, boundary, text))
	end
	width, height = attribute(page, "imageWidth"), attribute(page, "imageHeight")
	Page(image_path(folder, page["imageFilename"]), width === nothing ? 0 : parse(Int, width),
		height === nothing ? 0 : parse(Int, height), lines)
end

"""
	read_page(path) -> Page

The page of the ALTO or PAGE file at `path`: its image and its text lines, each with its baseline, boundary polygon and
transcription, read as Kraken reads them for a baseline model. A line without a baseline is left out. The image path is
the one the file gives, relative to the file's folder, or the file of the same name next to it when that path does not
exist.
"""
function read_page(path::AbstractString)
	root = EzXML.root(EzXML.readxml(path))
	name = EzXML.nodename(root)
	name == "alto" && return read_alto(root, dirname(path))
	name == "PcGts" && return read_page_xml(root, dirname(path))
	throw(ArgumentError("$(path) is neither an ALTO nor a PAGE file (its root element is $(name))."))
end

"""The text `s` escaped for an XML attribute or element."""
xml_escape(s::AbstractString) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

"""The points (x, y) flattened as ALTO writes them: `x0 y0 x1 y1 ...`."""
alto_points_text(points) = join((string(v) for p in points for v in p), " ")

"""The box of the points (x, y) as the ALTO attributes HPOS, VPOS, WIDTH and HEIGHT, after a space."""
function alto_box(points)
	xs, ys = first.(points), last.(points)
	" HPOS=\"$(minimum(xs))\" VPOS=\"$(minimum(ys))\" WIDTH=\"$(maximum(xs) - minimum(xs))\" HEIGHT=\"$(maximum(ys) - minimum(ys))\""
end
