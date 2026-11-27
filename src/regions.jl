# The regions of a page, the reading order of its lines and the whole of Kraken 7.2's segmenter (vectorize_regions and
# polygonal_reading_order in kraken/lib/segmentation.py, vec_regions, vec_lines and _segmentation_pred in
# kraken/lib/vgsl/spred.py), computed as Kraken computes them.

"""
	Region

A region of a page that the segmenter found: its identifier, its kind (`"text"` for blla) and its outline, points (x,
y) in pixels of the page.
"""
struct Region
	id::String
	kind::String
	boundary::Vector{NTuple{2, Int}}
end

"""
	Segmentation

The lines and regions of a page as the segmenter finds them: the page's image and size, its lines in reading order
(each with its baseline and outline, in pixels of the page, and no text), the kind of each line (`"default"` for
blla), its regions, and for each line the indices of the regions that hold it.
"""
struct Segmentation
	image::String
	width::Int
	height::Int
	lines::Vector{TextLine}
	kinds::Vector{String}
	regions::Vector{Region}
	line_regions::Vector{Vector{Int}}
end

function Base.show(io::IO, s::Segmentation)
	print(io, "Segmentation(", s.width, "×", s.height, ", ", length(s.lines), " lines, ", length(s.regions), " regions)")
end

"""The lines of `segmentation` as a `Page`, which `recognize` reads."""
Page(s::Segmentation) = Page(s.image, s.width, s.height, s.lines)

"""
The order in which GEOS 3.13's cascaded union takes `polygons`: the leaves of its STR tree (10 to a node), sorted by
the centres of their boxes along x, then within each vertical slice along y. The sorts are stable, as those of the
C++ library Shapely's GEOS is built with are for fewer than 30 items.
"""
function str_order(polygons; capacity::Int = 10)
	n = length(polygons)
	boxes = map(polygons) do polygon
		points = geos_coordinates(LibGEOS.exteriorRing(polygon))
		xs, ys = first.(points), last.(points)
		(minimum(xs) + maximum(xs), minimum(ys) + maximum(ys))
	end
	order = sort(collect(1:n); by = k -> boxes[k][1], alg = MergeSort)
	slices = unsafe_trunc(Int, ceil(sqrt(ceil(n / capacity))))
	per = unsafe_trunc(Int, ceil(n / slices))
	for start in 1:per:n
		stop = min(start + per - 1, n)
		order[start:stop] = sort(order[start:stop]; by = k -> boxes[k][2], alg = MergeSort)
	end
	order
end

"""
One step of GEOS 3.13's cascaded union: the union of `a` and `b` by OverlayNG, as its ClassicUnionStrategy makes it
(through `GEOSUnionPrec_r`, without the shortcut of `GEOSUnion_r` for geometries that can be combined as they are),
by a buffer of 0 where the overlay fails, its parts that are not polygons left out.
"""
function overlay_union(a, b)
	context = LibGEOS.get_context(a)
	result = LibGEOS.GEOSUnionPrec_r(context, a, b, 0.0)
	g = result == C_NULL ? LibGEOS.buffer(LibGEOS.GeometryCollection([a, b]), 0.0, 16) : LibGEOS.geomFromGEOS(result, context)
	g isa Union{LibGEOS.Polygon, LibGEOS.MultiPolygon} && return g
	g isa LibGEOS.GeometryCollection || return LibGEOS.createEmptyPolygon(context)
	parts = [p for p in geos_parts(g) if p isa LibGEOS.Polygon]
	LibGEOS.MultiPolygon([geos_coordinates(p) for p in parts])
end

"""
The union of `polygons` as GEOS 3.13's cascaded union makes it (Shapely's `unary_union` of polygons): taken in the
order of `str_order`, then joined in halves, each pair by GEOS's union, a single polygon kept as it is.
"""
function polygon_union(polygons)
	length(polygons) < 2 && return union_all(polygons)
	items = polygons[str_order(polygons)]
	function halves(start, stop)
		stop - start == 1 && return items[start]
		stop - start == 2 && return overlay_union(items[start], items[start + 1])
		middle = (start + stop) ÷ 2
		overlay_union(halves(start, middle), halves(middle, stop))
	end
	halves(1, length(items) + 1)
end

"""
Kraken's `vectorize_regions` of the map `heat` (width, height) of one kind of region: its blobs above `threshold`,
traced, joined where they overlap, and their outlines simplified by 10 pixels keeping their topology, as points (x, y)
at the scale of the map. Kraken stops with an error on a blob of a single pixel, which cannot be traced; such blobs
are left out.
"""
function vectorize_regions(heat::AbstractMatrix{Float32}; threshold::Float32 = 0.5f0)
	bin = permutedims(heat .> threshold)
	_, components = label_components(bin)
	polygons = LibGEOS.AbstractGeometry[]
	for pixels in components
		length(pixels) > 1 || continue
		b = boundary_tracing(pixels)
		length(b) > 2 && push!(polygons, geos_polygon(b))
	end
	merged = polygon_union(polygons)
	outlines = merged isa LibGEOS.Polygon ? [merged] : geos_parts(merged)
	map(outlines) do polygon
		simplified = LibGEOS.topologyPreserveSimplify(LibGEOS.boundary(polygon), 10.0)
		simplified isa Union{LibGEOS.LineString, LibGEOS.LinearRing} || error("A region has holes, which Kraken cannot write.")
		[(unsafe_trunc(Int, c), unsafe_trunc(Int, r)) for (r, c) in geos_points(simplified)]
	end
end

"""The points (x, y) scaled by `scale` (x, y) and cut to whole pixels, as Kraken's `scale_regions` and `scale_polygonal_lines`."""
scale_points(points, scale::NTuple{2, Float64}) = [(unsafe_trunc(Int, x * scale[1]), unsafe_trunc(Int, y * scale[2])) for (x, y) in points]

"""Kraken's `is_in_region`: whether `region` (a polygon) holds the middle of the line `line` (points), by its length."""
function is_in_region(line::Vector{NTuple{2, Int}}, region)
	ls = LibGEOS.LineString([[Float64(x), Float64(y)] for (x, y) in line])
	middle = LibGEOS.geomFromGEOS(LibGEOS.GEOSInterpolateNormalized_r(LibGEOS.get_context(ls), ls, 0.5), LibGEOS.get_context(ls))
	LibGEOS.contains(region, middle)
end

"""The box of the line `points` as Kraken orders lines by it: (rows from, to, columns from, to), Shapely's bounds."""
function line_box(points)
	xs, ys = Float64.(first.(points)), Float64.(last.(points))
	(minimum(ys), maximum(ys), minimum(xs), maximum(xs))
end

"""
Kraken's `_reading_order` of `boxes` (rows from, to, columns from, to): a matrix whose entry (i, j) is true when box
i comes before box j, by its rows where their columns overlap, and otherwise by its columns (from the left for
`text_direction` "lr", from the right for "rl") where no other box lies between them.
"""
function box_order(boxes::Vector{NTuple{4, Float64}}, text_direction::String = "lr")
	n = length(boxes)
	order = falses(n, n)
	overlaps(u, v) = u[3] < v[4] && u[4] > v[3]
	above(u, v) = u[1] < v[1]
	left_of(u, v) = u[4] < v[3]
	function separates(w, u, v)
		(w == u || w == v) && return false
		w[2] < min(u[1], v[1]) && return false
		w[1] > max(u[2], v[2]) && return false
		w[3] < u[4] && w[4] > v[3]
	end
	horizontal(u, v) = text_direction == "rl" ? !left_of(u, v) : left_of(u, v)
	for i in 1:n, j in 1:n
		u, v = boxes[i], boxes[j]
		if overlaps(u, v)
			above(u, v) && (order[i, j] = true)
		elseif !any(w -> separates(w, u, v), boxes) && horizontal(u, v)
			order[i, j] = true
		end
	end
	order
end

"""Kraken's `topsort` of the partial order `order`: each item after all that come before it, by depth-first search."""
function topsort(order::BitMatrix)
	n = size(order, 1)
	visited = falses(n)
	sorted = Int[]
	function visit(k)
		visited[k] && return
		visited[k] = true
		for i in findall(order[:, k])
			visit(i)
		end
		push!(sorted, k)
	end
	foreach(visit, 1:n)
	sorted
end

"""
Kraken's `polygonal_reading_order`: the order of the lines (their baselines, in pixels of the page), those in a region
ordered within it and the region ordered as a whole among the lines outside regions.
"""
function polygonal_reading_order(baselines::Vector{Vector{NTuple{2, Int}}}, regions::Vector{Vector{NTuple{2, Int}}};
		text_direction::String = "lr")
	polygons = [geos_polygon(r) for r in regions]
	region_lines = [Tuple{Int, NTuple{4, Float64}}[] for _ in regions]
	boxes = NTuple{4, Float64}[]
	keys = Tuple{Int, Symbol, Int}[]
	for (k, line) in enumerate(baselines)
		r = findfirst(p -> is_in_region(line, p), polygons)
		if r === nothing
			push!(boxes, line_box(line))
			push!(keys, (k, :line, k))
		else
			push!(region_lines[r], (k, line_box(line)))
		end
	end
	within = [Int[] for _ in regions]
	for (r, polygon) in enumerate(polygons)
		isempty(region_lines[r]) && continue
		sorted = topsort(box_order([b for (_, b) in region_lines[r]], text_direction))
		within[r] = [region_lines[r][i][1] for i in sorted]
		push!(boxes, line_box(regions[r]))
		# Kraken numbers a region after the last line, so that regions sort after the lines outside them
		push!(keys, (length(baselines) + r, :region, r))
	end
	ordered = Int[]
	for i in topsort(box_order(boxes, text_direction))
		_, kind, k = keys[i]
		kind == :line ? push!(ordered, k) : append!(ordered, within[k])
	end
	ordered
end

"""
	segment(model, image; path="", text_direction="horizontal-lr", torch_threads=1) -> Segmentation

The lines and regions of the page `image` (bytes, bands × width × height, as `read_image` reads it) found by the
segmentation model `model` (as `load_segmenter` loads it), as Kraken 7.2's `kraken segment -bl` finds them on the CPU:
the network's maps, the baselines of each kind found on them, the regions of each kind, the outline of each line, all
scaled to the page, and the lines in Kraken's reading order.

- `path`: the name of the page's image, written in the result.
- `text_direction`: as Kraken's, `"horizontal-lr"`, `"horizontal-rl"`, `"vertical-lr"` or `"vertical-rl"`: the
  direction of the lines, then that of the reading order.
- `torch_threads`: the number of PyTorch's threads to compute the network as, since one of blla's group normalisations
  sums in an order that depends on it. Kraken leaves it to PyTorch's default, which depends on the computer;
  `kraken --threads 1` sets it to 1.
"""
function segment(model::Segmenter, image::Array{UInt8, 3}; path::AbstractString = "",
		text_direction::AbstractString = "horizontal-lr", torch_threads::Int = 1)
	input, scaled = segmentation_input(model, image)
	maps = heatmaps(network_output(model, input; torch_threads), size(scaled)...)
	segment_maps(model, maps, scaled, size(image, 2), size(image, 3); path, text_direction)
end

"""
The lines and regions of `segment` from the network's maps `maps` (width, height, class, at the scale of the resized
page) and the resized page in grey `scaled`, for a page of `page_width` by `page_height` pixels.
"""
function segment_maps(model::Segmenter, maps::Array{Float32, 3}, scaled::Matrix{UInt8}, page_width::Int, page_height::Int;
		path::AbstractString = "", text_direction::AbstractString = "horizontal-lr")
	text_direction in ("horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl") ||
		throw(ArgumentError("The text direction is \"horizontal-lr\", \"horizontal-rl\", \"vertical-lr\" or \"vertical-rl\", not \"$(text_direction)\"."))
	axis, order_direction = split(text_direction, '-')
	width, height = size(scaled)
	mapping = model.class_mapping
	scale = (page_width / width, page_height / height)
	inverse = (1 / scale[1], 1 / scale[2])
	# the regions, at the scale of the page, and back at the scale of the maps as Kraken rounds them
	regions = Region[]
	for (kind, index) in sort(collect(get(mapping, "regions", Dict{String, Int}())); by = last)
		for outline in vectorize_regions(maps[:, :, index + 1])
			push!(regions, Region("region_$(length(regions) + 1)", kind, scale_points(outline, scale)))
		end
	end
	net_regions = [scale_points(r.boundary, inverse) for r in regions]
	region_polygons = [geos_polygon(r) for r in net_regions]
	# the baselines of each kind
	starts = maps[:, :, mapping["aux"]["_start_separator"] + 1]
	ends = maps[:, :, mapping["aux"]["_end_separator"] + 1]
	baselines = Tuple{String, Vector{NTuple{2, Int}}}[]
	for (kind, index) in sort(collect(mapping["baselines"]); by = last)
		for line in vectorize_lines(starts, ends, maps[:, :, index + 1]; text_direction = Symbol(axis))
			push!(baselines, (kind, line))
		end
	end
	# the outline of each line, bounded by the other lines and the regions that hold it
	features = image_features(scaled)
	# Kraken passes bool(topline): centre lines are outlined as baselines
	topline = model.topline === true
	found = Vector{Union{Nothing, Vector{NTuple{2, Int}}}}(undef, length(baselines))
	for k in eachindex(baselines)
		line = baselines[k][2]
		others = [b[2] for (j, b) in enumerate(baselines) if j != k]
		append!(others, [net_regions[r] for (r, p) in enumerate(region_polygons) if is_in_region(line, p)])
		found[k] = line_outline(line, others, features; topline)
	end
	kept = [k for k in eachindex(baselines) if found[k] !== nothing]
	page_baselines = [scale_points(baselines[k][2], scale) for k in kept]
	page_outlines = [scale_points(found[k], scale) for k in kept]
	kinds = [baselines[k][1] for k in kept]
	page_regions = [geos_polygon(r.boundary) for r in regions]
	line_regions = [[r for (r, p) in enumerate(page_regions) if is_in_region(b, p)] for b in page_baselines]
	order = isempty(kept) ? Int[] : polygonal_reading_order(page_baselines, [r.boundary for r in regions]; text_direction = String(order_direction))
	lines = [TextLine("line_$(i)", page_baselines[k], page_outlines[k], "") for (i, k) in enumerate(order)]
	Segmentation(path, page_width, page_height, lines, kinds[order], regions, line_regions[order])
end

"""
The entities of Kraken's serializer for `s`, in order: a region (index > 0) with the lines it holds, or a line outside
regions (index < 0). A line goes where the previous one went unless its first region is not the previous line's, so a
line outside regions sends the lines after it out of their region too, until one starts another region; the regions
without lines come last.
"""
function kraken_entities(s::Segmentation)
	entities = Tuple{Int, Vector{Int}}[]
	used = falses(length(s.regions))
	current = 0
	previous_region = 0
	for k in eachindex(s.lines)
		regions = s.line_regions[k]
		if isempty(regions)
			current = 0
		elseif previous_region != regions[1]
			previous_region = regions[1]
			used[previous_region] = true
			push!(entities, (previous_region, Int[]))
			current = length(entities)
		end
		current == 0 ? push!(entities, (-k, Int[])) : push!(entities[current][2], k)
	end
	append!(entities, [(r, Int[]) for r in eachindex(s.regions) if !used[r]])
end

"""
	write_alto(path, segmentation) -> path

Writes the lines and regions of `segmentation` (as `segment` finds them) to an ALTO 4 file at `path`, laid out as
Kraken 7 lays out the ALTO of a segmentation: the kinds of lines and regions as tags, the reading order of the lines,
then each line in that order with its box, baseline and outline, inside the TextBlock of its first region, the lines
outside regions in TextBlocks of their own, and last the regions that hold no line. `read_page` reads the file back.
"""
function write_alto(path::AbstractString, s::Segmentation)
	typology = sort(unique(vcat([("type", k) for k in s.kinds], [("type", r.kind) for r in s.regions])))
	tag(kind) = "TYPE_$(findfirst(==(("type", kind)), typology))"
	entities = kraken_entities(s)

	io = IOBuffer()
	print(io, """
	<?xml version="1.0" encoding="UTF-8"?>
	<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	    xmlns="http://www.loc.gov/standards/alto/ns-v4#"
	    xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-3.xsd">
	    <Description>
	        <MeasurementUnit>pixel</MeasurementUnit>
	        <sourceImageInformation>
	            <fileName>$(xml_escape(s.image))</fileName>
	        </sourceImageInformation>
	        <Processing ID="OCR_0">
	            <processingCategory>contentGeneration</processingCategory>
	            <processingStepDescription>Baseline and region segmentation</processingStepDescription>
	            <processingSoftware>
	                <softwareName>MedievalATR.jl</softwareName>
	                <softwareVersion>$(pkgversion(@__MODULE__))</softwareVersion>
	            </processingSoftware>
	        </Processing>
	    </Description>
	    <Tags>
	""")
	for (k, (type, label)) in enumerate(typology)
		println(io, "        <OtherTag DESCRIPTION=\"\" ID=\"TYPE_$(k)\" TYPE=\"$(type)\" LABEL=\"$(xml_escape(label))\"/>")
	end
	println(io, "    </Tags>")
	if !isempty(s.lines)
		println(io, "    <ReadingOrder>\n        <OrderedGroup ID=\"ro_0\">")
		for (k, line) in enumerate(s.lines)
			println(io, "            <ElementRef ID=\"o_$(k)\" REF=\"$(xml_escape(line.id))\"/>")
		end
		println(io, "        </OrderedGroup>\n    </ReadingOrder>")
	end
	println(io, "    <Layout>\n        <Page WIDTH=\"$(s.width)\" HEIGHT=\"$(s.height)\" PHYSICAL_IMG_NR=\"0\" ID=\"page_0\">")
	println(io, "            <PrintSpace HPOS=\"0\" VPOS=\"0\" WIDTH=\"$(s.width)\" HEIGHT=\"$(s.height)\">")
	function shape(points, indent)
		println(io, indent, "<Shape>\n", indent, "    <Polygon POINTS=\"", alto_points_text(points), "\"/>\n", indent, "</Shape>")
	end
	function write_line(k)
		line = s.lines[k]
		outline = line.boundary === nothing || isempty(line.boundary) ? nothing : line.boundary
		println(io, "                    <TextLine ID=\"$(xml_escape(line.id))\"", outline === nothing ? "" : alto_box(outline),
			" BASELINE=\"$(alto_points_text(line.baseline))\" TAGREFS=\"$(tag(s.kinds[k]))\">")
		outline === nothing || shape(outline, " "^24)
		println(io, "                        <String CONTENT=\"$(xml_escape(line.text))\"/>\n                    </TextLine>")
	end
	in_loose_block = false
	for (e, (index, lines)) in enumerate(entities)
		if index > 0
			in_loose_block && println(io, "                </TextBlock>")
			in_loose_block = false
			region = s.regions[index]
			outline = isempty(region.boundary) ? nothing : region.boundary
			println(io, "                <TextBlock ID=\"$(xml_escape(region.id))\"", outline === nothing ? "" : alto_box(outline),
				" TAGREFS=\"$(tag(region.kind))\">")
			outline === nothing || shape(outline, " "^20)
			foreach(write_line, lines)
			println(io, "                </TextBlock>")
		else
			in_loose_block || println(io, "                <TextBlock ID=\"textblock_$(e)\">")
			in_loose_block = true
			write_line(-index)
		end
	end
	in_loose_block && println(io, "                </TextBlock>")
	print(io, "            </PrintSpace>\n        </Page>\n    </Layout>\n</alto>")
	write(path, take!(io))
	path
end

"""The points (x, y) as PAGE writes them: `x0,y0 x1,y1 ...`."""
page_points_text(points) = join(("$(p[1]),$(p[2])" for p in points), " ")

"""
	write_page_xml(path, segmentation) -> path

Writes the lines and regions of `segmentation` (as `segment` finds them) to a PAGE file (2019 schema) at `path`, laid
out as Kraken 7 lays out the PAGE file of a segmentation: each line in reading order with its outline and baseline,
inside the TextRegion of its first region, the lines outside regions in TextRegions that cover the page, and last the
regions that hold no line; the kind of each line and region in its `custom` attribute (`type {type:text;}`). A line
without an outline is left out, as Kraken leaves it out. `read_page` reads the file back.
"""
function write_page_xml(path::AbstractString, s::Segmentation)
	custom(kind) = "type {type:$(xml_escape(kind));}"
	date = Dates.format(Dates.now(Dates.UTC), Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "+00:00"
	io = IOBuffer()
	print(io, """
	<?xml version="1.0" encoding="UTF-8"?>
	<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd">
	    <Metadata>
	        <Creator>MedievalATR.jl $(pkgversion(@__MODULE__))</Creator>
	        <Created>$(date)</Created>
	        <LastChange>$(date)</LastChange>
	    </Metadata>
	    <Page imageFilename="$(xml_escape(s.image))" imageWidth="$(s.width)" imageHeight="$(s.height)">
	""")
	function write_line(k)
		line = s.lines[k]
		(line.boundary === nothing || isempty(line.boundary)) && return
		println(io, "            <TextLine id=\"$(xml_escape(line.id))\" custom=\"$(custom(s.kinds[k]))\">")
		println(io, "                <Coords points=\"$(page_points_text(line.boundary))\"/>")
		isempty(line.baseline) || println(io, "                <Baseline points=\"$(page_points_text(line.baseline))\"/>")
		isempty(line.text) || println(io, "                <TextEquiv><Unicode>$(xml_escape(line.text))</Unicode></TextEquiv>")
		println(io, "            </TextLine>")
	end
	entities = kraken_entities(s)
	in_loose_region = false
	for (e, (index, lines)) in enumerate(entities)
		if index > 0
			in_loose_region && println(io, "        </TextRegion>")
			in_loose_region = false
			region = s.regions[index]
			println(io, "        <TextRegion id=\"$(xml_escape(region.id))\" custom=\"$(custom(region.kind))\">")
			println(io, "            <Coords points=\"", isempty(region.boundary) ? "0,0 0,0 0,0" : page_points_text(region.boundary), "\"/>")
			foreach(write_line, lines)
			println(io, "        </TextRegion>")
		else
			if !in_loose_region
				println(io, "        <TextRegion id=\"textblock_$(e)\">")
				println(io, "            <Coords points=\"0,0 0,$(s.height) $(s.width),$(s.height) $(s.width),0\"/>")
			end
			in_loose_region = true
			write_line(-index)
		end
	end
	in_loose_region && println(io, "        </TextRegion>")
	print(io, "    </Page>\n</PcGts>")
	write(path, take!(io))
	path
end
