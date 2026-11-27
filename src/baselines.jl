# The baselines of a page from the segmenter's maps, as Kraken 7.2 finds them (vectorize_lines in
# kraken/lib/segmentation.py): the baseline map put through the ridge filter and thresholded, thinned to a skeleton, the
# skeleton's end points joined by minimum-cost paths along it, each path simplified, extended to the end of its blob and
# turned to run from the start of the line to its end. Each step is computed as scikit-image 0.26, SciPy 1.16 and
# Shapely 2.1 (GEOS 3.13.1) compute it, so that the same lines come out.
#
# A grid here is a matrix (rows, columns), as NumPy holds an image, and a point (row, column) from 0, so that the flat
# index of a pixel, row + column * rows, is the one scikit-image's path search orders pixels by.

# scikit-image's table of the eight neighbours of a pixel (_fast_skeletonize): 1 and 3 are removed in the first pass of
# Zhang and Suen's thinning, 2 and 3 in the second
const ZHANG_SUEN_TABLE = UInt8[
	0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
	0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
	0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

"""
scikit-image's `skeletonize` of the grid `image` (Zhang and Suen's thinning): the pixels each pass may remove are
found on the image as it was at the start of the pass, until a whole round removes none.
"""
function skeletonize(image::AbstractMatrix{Bool})
	nrows, ncols = size(image)
	skeleton = zeros(UInt8, nrows + 2, ncols + 2)
	skeleton[2:nrows + 1, 2:ncols + 1] .= image
	cleaned = copy(skeleton)
	removed = true
	while removed
		removed = false
		for pass in 1:2
			@inbounds for col in 2:ncols + 1, row in 2:nrows + 1
				skeleton[row, col] == 0 && continue
				code = skeleton[row - 1, col - 1] + 2 * skeleton[row - 1, col] + 4 * skeleton[row - 1, col + 1] +
					8 * skeleton[row, col + 1] + 16 * skeleton[row + 1, col + 1] + 32 * skeleton[row + 1, col] +
					64 * skeleton[row + 1, col - 1] + 128 * skeleton[row, col - 1]
				kind = ZHANG_SUEN_TABLE[code + 1]
				if kind == 3 || (kind == 1 && pass == 1) || (kind == 2 && pass == 2)
					cleaned[row, col] = 0
					removed = true
				end
			end
			copyto!(skeleton, cleaned)
		end
	end
	skeleton[2:nrows + 1, 2:ncols + 1] .!= 0
end

"""The pixels of the grid `skeleton` with exactly one of their eight neighbours set, in NumPy's row order."""
function skeleton_endpoints(skeleton::AbstractMatrix{Bool})
	nrows, ncols = size(skeleton)
	points = NTuple{2, Int}[]
	@inbounds for col in 1:ncols, row in 1:nrows
		skeleton[row, col] || continue
		neighbours = 0
		for dc in -1:1, dr in -1:1
			(dr == 0 && dc == 0) && continue
			r, c = row + dr, col + dc
			if 1 <= r <= nrows && 1 <= c <= ncols && skeleton[r, c]
				neighbours += 1
			end
		end
		neighbours == 1 && push!(points, (row - 1, col - 1))
	end
	sort!(points)
end

"""
The connected components of the grid `image`, its pixels joined to their eight neighbours, as SciPy's `label` numbers
them, by their first pixel in row order: the matrix of labels (0 for the background) and the pixels of each component,
in row order, as scikit-image's `regionprops` gives them.
"""
function label_components(image::AbstractMatrix{Bool})
	nrows, ncols = size(image)
	labels = zeros(Int, nrows, ncols)
	count = 0
	stack = NTuple{2, Int}[]
	@inbounds for row in 1:nrows, col in 1:ncols
		(image[row, col] && labels[row, col] == 0) || continue
		count += 1
		labels[row, col] = count
		push!(stack, (row, col))
		while !isempty(stack)
			r, c = pop!(stack)
			for dc in -1:1, dr in -1:1
				rr, cc = r + dr, c + dc
				if 1 <= rr <= nrows && 1 <= cc <= ncols && image[rr, cc] && labels[rr, cc] == 0
					labels[rr, cc] = count
					push!(stack, (rr, cc))
				end
			end
		end
	end
	pixels = [NTuple{2, Int}[] for _ in 1:count]
	@inbounds for row in 1:nrows, col in 1:ncols
		labels[row, col] > 0 && push!(pixels[labels[row, col]], (row - 1, col - 1))
	end
	labels, pixels
end

"""
The end points of `skeleton` that Kraken keeps when there are more than `max_endpoints`: those of the components of
the skeleton with at most 10 end points, and, if these are still too many, those of the largest such components that
fit in `max_endpoints`. Components of the same size are taken in the order of their labels.
"""
function filter_endpoints(skeleton::AbstractMatrix{Bool}, points::Vector{NTuple{2, Int}}, max_endpoints::Int)
	length(points) > max_endpoints || return points
	labels, pixels = label_components(skeleton)
	components = [labels[r + 1, c + 1] for (r, c) in points]
	counts = Dict{Int, Int}()
	for k in components
		counts[k] = get(counts, k, 0) + 1
	end
	valid = sort!([k for (k, n) in counts if n <= 10])
	if sum(counts[k] for k in valid; init = 0) > max_endpoints
		budget = 0
		kept = Int[]
		for k in sort(valid; by = k -> length(pixels[k]), rev = true)
			budget + counts[k] > max_endpoints && break
			push!(kept, k)
			budget += counts[k]
		end
		valid = kept
	end
	keep = Set(valid)
	points[[k in keep for k in components]]
end

"""
scikit-image's `FastUpdateBinaryHeap`: a binary heap of values, each with a reference that is pushed only once, its
value lowered in place when pushed again. Indices are those of scikit-image (from 0), so that ties come out in the same
order.
"""
mutable struct PathHeap
	levels::Int
	min_levels::Int
	count::Int
	values::Vector{Float64}
	references::Vector{Int}
	crossref::Vector{Int}
	pushed::Bool
	popped::Int
end

function PathHeap(capacity::Int, max_reference::Int)
	levels = 0
	while 2^levels < capacity
		levels += 1
	end
	PathHeap(levels, levels, 0, fill(Inf, 2 * 2^levels), fill(-1, 2^levels), fill(-1, max_reference + 1), false, -1)
end

function heap_update!(heap::PathHeap)
	values = heap.values
	for level in heap.levels:-1:2
		i0 = (1 << level) - 1
		for i in i0:2:i0 + (1 << level) - 1
			ii = (i - 1) ÷ 2
			values[ii + 1] = values[i + 1] < values[i + 2] ? values[i + 1] : values[i + 2]
		end
	end
end

function heap_update_one!(heap::PathHeap, i::Int)
	values = heap.values
	iseven(i) && (i -= 1)
	for _ in heap.levels:-1:2
		ii = (i - 1) ÷ 2
		values[ii + 1] = values[i + 1] < values[i + 2] ? values[i + 1] : values[i + 2]
		i = isodd(ii) ? ii : ii - 1
	end
end

function heap_resize!(heap::PathHeap, change::Int)
	levels = heap.levels + change
	number = 2^levels
	values = fill(Inf, 2number)
	references = fill(-1, number)
	if heap.count > 0
		i1, i2 = number - 1, 2^heap.levels - 1
		n = min(number, 2^heap.levels)
		for i in 0:n - 1
			values[i1 + i + 1] = heap.values[i2 + i + 1]
			references[i + 1] = heap.references[i + 1]
		end
	end
	heap.values, heap.references, heap.levels = values, references, levels
	heap_update!(heap)
end

function heap_remove!(heap::PathHeap, i1::Int)
	levels, count = heap.levels, heap.count
	i0 = (1 << levels) - 1
	i2 = i0 + count - 1
	r1, r2 = i1 - i0, count - 1
	heap.crossref[heap.references[r2 + 1] + 1] = r1
	heap.crossref[heap.references[r1 + 1] + 1] = -1
	heap.values[i1 + 1] = heap.values[i2 + 1]
	heap.references[r1 + 1] = heap.references[r2 + 1]
	heap.values[i2 + 1] = Inf
	heap.count -= 1
	count -= 1
	if levels > heap.min_levels && count < (1 << (levels - 2))
		heap_resize!(heap, -1)
	else
		heap_update_one!(heap, i1)
		heap_update_one!(heap, i2)
	end
end

function heap_append!(heap::PathHeap, value::Float64, reference::Int)
	if heap.count >= (1 << heap.levels)
		heap_resize!(heap, 1)
	end
	count = heap.count
	i = (1 << heap.levels) - 1 + count
	heap.values[i + 1] = value
	heap.references[count + 1] = reference
	heap.count += 1
	heap_update_one!(heap, i)
	heap.crossref[reference + 1] = count
end

"""`push_fast`: append `reference` with `value`, or set its value if it is in the heap."""
function heap_push!(heap::PathHeap, value::Float64, reference::Int)
	ir = heap.crossref[reference + 1]
	if ir != -1
		i = (1 << heap.levels) - 1 + ir
		heap.values[i + 1] = value
		heap_update_one!(heap, i)
	else
		heap_append!(heap, value, reference)
	end
end

"""`push_if_lower_fast`: append `reference`, or lower its value; `heap.pushed` says whether either was done."""
function heap_push_if_lower!(heap::PathHeap, value::Float64, reference::Int)
	ir = heap.crossref[reference + 1]
	heap.pushed = true
	if ir != -1
		i = (1 << heap.levels) - 1 + ir
		if heap.values[i + 1] > value
			heap.values[i + 1] = value
			heap_update_one!(heap, i)
		else
			heap.pushed = false
		end
	else
		heap_append!(heap, value, reference)
	end
end

"""`pop_fast`: the smallest value, its reference in `heap.popped`, the left one of equal values first."""
function heap_pop!(heap::PathHeap)
	values = heap.values
	i = 1
	for _ in 1:heap.levels - 1
		i = values[i + 1] <= values[i + 2] ? 2i + 1 : 2(i + 1) + 1
	end
	values[i + 1] <= values[i + 2] || (i += 1)
	value = values[i + 1]
	heap.popped = heap.references[i - ((1 << heap.levels) - 1) + 1]
	heap.count > 0 && heap_remove!(heap, i)
	value
end

# scikit-image's eight steps from a pixel (make_offsets), in (row, column)
const PATH_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

"""
Kraken's `LineMCP` on the grid `skeleton` from the end points `starts`: fronts grown from every end point at once along
the skeleton (scikit-image's MCP_Connect, each step costing nothing on the skeleton and one off it), stopped as soon as
a front would leave the skeleton. Where two fronts meet, the end points they came from are joined, by the first meeting
found; the paths come in the order the pairs first met, each from one end point to the other.
"""
function skeleton_paths(skeleton::AbstractMatrix{Bool}, starts::Vector{NTuple{2, Int}})
	nrows, ncols = size(skeleton)
	size_ = nrows * ncols
	costs = [s ? 0.0 : 1.0 for s in vec(skeleton)]
	cumulative = fill(Inf, size_)
	traceback = fill(Int8(-2), size_)
	origin = fill(-1, size_)
	heap = PathHeap(128, size_ - 1)
	for (id, (r, c)) in enumerate(starts)
		flat = r + c * nrows
		traceback[flat + 1] = -1
		heap_push!(heap, costs[flat + 1], flat)
		origin[flat + 1] = id - 1
	end
	lengths = [sqrt(Float64(dr^2 + dc^2)) for (dr, dc) in PATH_STEPS]
	meetings = Dict{NTuple{2, Int}, Tuple{NTuple{2, Int}, NTuple{2, Int}}}()
	scores = Dict{NTuple{2, Int}, Float64}()
	order = NTuple{2, Int}[]
	for _ in 1:size_
		heap.count == 0 && break
		cumcost = heap_pop!(heap)
		index = heap.popped
		cumulative[index + 1] = cumcost
		# LineMCP.goal_reached: the search ends at the first pixel off the skeleton
		cumcost != 0 && break
		r, c = index % nrows, index ÷ nrows
		for (i, (dr, dc)) in enumerate(PATH_STEPS)
			rr, cc = r + dr, c + dc
			(0 <= rr < nrows && 0 <= cc < ncols) || continue
			new = rr + cc * nrows
			if cumulative[new + 1] != Inf
				id1, id2 = origin[index + 1], origin[new + 1]
				if id1 >= 0 && id2 >= 0 && id1 != id2
					key = (min(id1, id2), max(id1, id2))
					s = cumulative[index + 1] + cumulative[new + 1]
					if get(scores, key, Inf) > s
						haskey(meetings, key) || push!(order, key)
						meetings[key] = ((r, c), (rr, cc))
						scores[key] = s
					end
				end
				continue
			end
			cost, new_cost = costs[index + 1], costs[new + 1]
			(new_cost < 0 || new_cost == Inf) && continue
			new_cumcost = cumcost + lengths[i] * 0.5 * (cost + new_cost)
			if new_cumcost != Inf
				heap_push_if_lower!(heap, new_cumcost, new)
				if heap.pushed
					traceback[new + 1] = i - 1
					origin[new + 1] = origin[index + 1]
				end
			end
		end
	end
	function trace(point)
		path = [point]
		flat = point[1] + point[2] * nrows
		while traceback[flat + 1] != -1
			dr, dc = PATH_STEPS[traceback[flat + 1] + 1]
			point = (point[1] - dr, point[2] - dc)
			flat -= dr + dc * nrows
			push!(path, point)
		end
		reverse!(path)
	end
	[vcat(trace(meetings[key][1]), reverse(trace(meetings[key][2]))) for key in order]
end

# Moore's neighbourhood, clockwise from the pixel above, in (row, column)
const MOORE_STEPS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

"""
Kraken's `boundary_tracing` of a component (its pixels in row order): the pixels of its outline, traced clockwise by
Moore's neighbourhood from its first pixel that is not isolated.
"""
function boundary_tracing(pixels::Vector{NTuple{2, Int}})
	minr, minc = minimum(first, pixels), minimum(last, pixels)
	maxr, maxc = maximum(first, pixels), maximum(last, pixels)
	# padded by one pixel; the trace works on (row, column) from 0, binary[r + 1, c + 1]
	binary = falses(maxr - minr + 3, maxc - minc + 3)
	ys = [r - minr + 1 for (r, _) in pixels]
	xs = [c - minc + 1 for (_, c) in pixels]
	for (y, x) in zip(ys, xs)
		binary[y + 1, x + 1] = true
	end
	set(y, x) = binary[y + 1, x + 1]
	k = 1
	start = (0, 0)
	while true
		k > length(pixels) && error("The component has only isolated pixels.")
		start = (ys[k], xs[k])
		sum(set(start[1] + dy, start[2] + dx) for dy in -1:1, dx in -1:1) > 1 && break
		k += 1
	end
	backtrack_start = !set(start[1] + 1, start[2]) && !set(start[1] + 1, start[2] - 1) ?
		(start[1] + 1, start[2]) : (start[1], start[2] - 1)
	current, backtrack = start, backtrack_start
	boundary = NTuple{2, Int}[]
	for _ in 1:8 * length(binary)
		neighbours = [(current[1] + dy, current[2] + dx) for (dy, dx) in MOORE_STEPS]
		i = findfirst(==(backtrack), neighbours)
		i === nothing && error("The backtrack pixel is not a neighbour.")
		neighbours = vcat(neighbours[i:end], neighbours[1:i - 1])
		idx = something(findfirst(p -> set(p...), neighbours), 1)
		push!(boundary, current)
		backtrack = neighbours[idx == 1 ? end : idx - 1]
		current = neighbours[idx]
		if current == start && backtrack == backtrack_start
			return [(y + minr - 1, x + minc - 1) for (y, x) in boundary]
		end
	end
	error("The boundary trace does not close.")
end

geos_point(p) = LibGEOS.Point(Float64(p[1]), Float64(p[2]))
geos_line(points) = LibGEOS.LineString([[Float64(p[1]), Float64(p[2])] for p in points])

"""
The point where the outline of `polygon` crosses the segment from `point` back ten times the step from `next` to
`point`, cut to whole pixels as NumPy casts to integers, or `point` itself where the segment crosses it at none or more
than one point (its nearest point on `polygon`, inside which it lies).
"""
function boundary_end(polygon, point::NTuple{2, Int}, next::NTuple{2, Int})
	far = (point[1] - 10 * (next[1] - point[1]), point[2] - 10 * (next[2] - point[2]))
	crossing = LibGEOS.intersection(LibGEOS.boundary(polygon), geos_line([far, point]))
	if crossing isa LibGEOS.Point
		x, y = LibGEOS.getGeomX(crossing), LibGEOS.getGeomY(crossing)
	else
		nearest = LibGEOS.nearestPoints(geos_point(point), polygon)[2]
		x, y = LibGEOS.getGeomX(nearest), LibGEOS.getGeomY(nearest)
	end
	(unsafe_trunc(Int, x), unsafe_trunc(Int, y))
end

"""
Kraken's `_extend_boundaries`: each line of `lines` (points (row, column)) whose ends lie inside the outline of the
blob of `bin` that holds it is extended at each end to that outline. The outlines are traced from the blobs of 6
pixels or more, simplified by 0.01 pixels keeping their topology, and repaired by a buffer of 0, as Shapely makes them.
"""
function extend_boundaries!(lines::Vector{Vector{NTuple{2, Int}}}, bin::AbstractMatrix{Bool})
	_, components = label_components(bin)
	outlines = LibGEOS.AbstractGeometry[]
	for pixels in components
		length(pixels) < 6 && continue
		try
			b = boundary_tracing(pixels)
			if length(b) > 3
				ring = [[Float64(r), Float64(c)] for (r, c) in b]
				ring[1] == ring[end] || push!(ring, ring[1])
				polygon = LibGEOS.Polygon([ring])
				push!(outlines, LibGEOS.buffer(LibGEOS.topologyPreserveSimplify(polygon, 0.01), 0.0, 16))
			end
		catch
			continue
		end
	end
	for line in lines
		polygon = nothing
		try
			ls = geos_line(line)
			k = findfirst(o -> LibGEOS.contains(o, ls), outlines)
			k === nothing && continue
			polygon = outlines[k]
		catch
			continue
		end
		if LibGEOS.contains(polygon, geos_point(line[1]))
			line[1] = boundary_end(polygon, line[1], line[2])
		end
		if LibGEOS.contains(polygon, geos_point(line[end]))
			line[end] = boundary_end(polygon, line[end], line[end - 1])
		end
	end
	lines
end

"""
The largest value of `map` (width, height) within the 20 by 20 window of SciPy's `maximum_filter` (size 20, mode
"reflect") at the point (row, column), from 10 before it to 9 after it in each direction.
"""
function window_maximum(map::AbstractMatrix{Float32}, point::NTuple{2, Int})
	width, height = size(map)
	r, c = point
	m = -Inf32
	@inbounds for y in max(r - 10, 0):min(r + 9, height - 1), x in max(c - 10, 0):min(c + 9, width - 1)
		m = max(m, map[x + 1, y + 1])
	end
	m
end

"""
Kraken's orientation of the lines (points (row, column)): a line runs from the end where the map of line starts is
higher than that of line ends by more than 0.2 to the end where it is lower by more than 0.2; when the maps do not say,
a horizontal line runs upwards (from its lower end, by rows) and a vertical one downwards. Lines shorter than
`min_length` pixels are left out, and the points of the others are given as (x, y).
"""
function orient_lines(lines::Vector{Vector{NTuple{2, Int}}}, starts::AbstractMatrix{Float32}, ends::AbstractMatrix{Float32};
		text_direction::Symbol = :horizontal, min_length::Real = 5)
	text_direction in (:horizontal, :vertical) || error("The text direction is :horizontal or :vertical.")
	limit = 0.2f0
	oriented = Vector{NTuple{2, Int}}[]
	for line in lines
		left = window_maximum(starts, line[1]) - window_maximum(ends, line[1])
		right = window_maximum(starts, line[end]) - window_maximum(ends, line[end])
		if left > limit && right < -limit
		elseif left < -limit && right > limit
			line = reverse(line)
		elseif text_direction == :horizontal
			line[1][2] > line[end][2] && (line = reverse(line))
		else
			line[1][1] > line[end][1] && (line = reverse(line))
		end
		len = 0.0
		for k in 2:length(line)
			dy, dx = Float64(line[k][1] - line[k - 1][1]), Float64(line[k][2] - line[k - 1][2])
			len += sqrt(dx * dx + dy * dy)
		end
		len >= min_length && push!(oriented, [(c, r) for (r, c) in line])
	end
	oriented
end

"""
	vectorize_lines(starts, ends, baselines; threshold=0.17, min_length=5, text_direction=:horizontal,
		max_endpoints=400) -> Vector{Vector{NTuple{2, Int}}}

Kraken's `vectorize_lines`: the baselines (points (x, y) from 0, at the scale of the maps) found on the maps of line
starts, line ends and one kind of baseline (width, height), as Kraken 7.2 finds them. `stages`, when given, a
`Dict{Symbol, Any}`, is filled with the map after the ridge filter thresholded (`:bin`), its skeleton, its end points,
the paths between them, these simplified and extended.
"""
function vectorize_lines(starts::AbstractMatrix{Float32}, ends::AbstractMatrix{Float32}, baselines::Matrix{Float32};
		threshold::Float32 = 0.17f0, min_length::Real = 5, text_direction::Symbol = :horizontal, max_endpoints::Int = 400,
		stages::Union{Dict{Symbol, Any}, Nothing} = nothing)
	bin = permutedims(sato(baselines) .> threshold)
	vectorize_bin(starts, ends, bin; min_length, text_direction, max_endpoints, stages)
end

"""
The baselines of `vectorize_lines` from the thresholded ridge map `bin` (a grid, rows by columns), the steps after the
ridge filter.
"""
function vectorize_bin(starts::AbstractMatrix{Float32}, ends::AbstractMatrix{Float32}, bin::AbstractMatrix{Bool};
		min_length::Real = 5, text_direction::Symbol = :horizontal, max_endpoints::Int = 400,
		stages::Union{Dict{Symbol, Any}, Nothing} = nothing)
	skeleton = skeletonize(bin)
	points = filter_endpoints(skeleton, skeleton_endpoints(skeleton), max_endpoints)
	record(name, value) = stages === nothing || (stages[name] = deepcopy(value))
	record(:bin, bin)
	record(:skeleton, skeleton)
	record(:extrema, points)
	length(points) < 2 && return Vector{NTuple{2, Int}}[]
	paths = skeleton_paths(skeleton, points)
	record(:connections, paths)
	lines = [[(Int(r), Int(c)) for (r, c) in approximate_polygon([(Float64(r), Float64(c)) for (r, c) in path], 3.0)]
		for path in paths]
	record(:approximated, lines)
	extend_boundaries!(lines, bin)
	record(:extended, lines)
	orient_lines(lines, starts, ends; text_direction, min_length)
end
