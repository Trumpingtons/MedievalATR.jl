# The outline of each line, as Kraken 7.2 computes it (calculate_polygonal_environment in kraken/lib/segmentation.py):
# the region above and below the baseline that the line may take, bounded by the page and by the other lines and
# regions, then on each side the seam of least image features through that region, cut as a seam carver cuts an image.
# Each step is computed as NumPy 2.4, SciPy 1.16, scikit-image 0.26, Pillow 13 and Shapely 2.1 (GEOS 3.13.1) compute it.
#
# Points are (x, y) at the scale of the network's maps, the features a grid (rows, columns) as `image_features` makes it.

geos_coordinates(g) = LibGEOS.GeoInterface.coordinates(g)
geos_points(g) = [(c[1], c[2]) for c in geos_coordinates(g)]

"""A Shapely polygon of `points`, closed as Shapely closes it, when its last point is not its first."""
function geos_polygon(points)
	ring = [[Float64(p[1]), Float64(p[2])] for p in points]
	ring[1] == ring[end] || push!(ring, ring[1])
	length(ring) >= 4 || error("A polygon needs at least three points.")
	LibGEOS.Polygon([ring])
end

"""Shapely's `unary_union` of `geometries`: GEOS's union of the collection of them."""
union_all(geometries) = LibGEOS.unaryUnion(LibGEOS.GeometryCollection(collect(geometries)))

"""Shapely's `offset_curve` of `line` by `distance` (left for positive, right for negative), with round joins of 16 segments."""
function offset_curve(line, distance::Float64)
	context = LibGEOS.get_context(line)
	result = LibGEOS.GEOSOffsetCurve_r(context, line, distance, Int32(16), Int32(1), 5.0)
	result == C_NULL && error("LibGEOS: Error in GEOSOffsetCurve")
	LibGEOS.geomFromGEOS(result, context)
end

"""The parts of `geometry`, as Shapely's `get_parts` gives them: the geometry itself unless it is a collection."""
geos_parts(g) = g isa Union{LibGEOS.MultiLineString, LibGEOS.MultiPoint, LibGEOS.MultiPolygon, LibGEOS.GeometryCollection} ?
	[LibGEOS.getGeometry(g, k) for k in 1:LibGEOS.numGeometries(g)] : [g]

"""NumPy's sum of `v` along a strided axis: in order, from its first value."""
strided_sum(v) = foldl(+, v)

"""NumPy's `around` to whole numbers: halves to the even number."""
around(x::Float64) = round(Int, x, RoundNearest)

"""scikit-image's `draw.line` from (`r0`, `c0`) to (`r1`, `c1`): the pixels of Bresenham's line, as rows and columns."""
function draw_line(r0::Int, c0::Int, r1::Int, c1::Int)
	steep = false
	r, c = r0, c0
	dr, dc = abs(r1 - r0), abs(c1 - c0)
	sc = c1 - c > 0 ? 1 : -1
	sr = r1 - r > 0 ? 1 : -1
	if dr > dc
		steep = true
		c, r = r, c
		dc, dr = dr, dc
		sc, sr = sr, sc
	end
	d = 2dr - dc
	rr = zeros(Int, max(dc, dr) + 1)
	cc = zeros(Int, max(dc, dr) + 1)
	for i in 1:dc
		if steep
			rr[i], cc[i] = c, r
		else
			rr[i], cc[i] = r, c
		end
		while d >= 0
			r += sr
			d -= 2dc
		end
		c += sc
		d += 2dr
	end
	rr[dc + 1], cc[dc + 1] = r1, c1
	rr, cc
end

"""
SciPy's `distance_transform_cdt` with its default chessboard metric: for each pixel of `mask` that is set, the number
of king's moves to the nearest pixel that is not (-1 where there is none), by two passes over the grid.
"""
function chessboard_distance(mask::AbstractMatrix{Bool})
	nrows, ncols = size(mask)
	far = nrows + ncols + 1
	d = [m ? far : 0 for m in mask]
	@inbounds for r in 1:nrows, c in 1:ncols
		d[r, c] == 0 && continue
		best = d[r, c]
		for (dr, dc) in ((-1, -1), (-1, 0), (-1, 1), (0, -1))
			rr, cc = r + dr, c + dc
			if 1 <= rr <= nrows && 1 <= cc <= ncols
				best = min(best, d[rr, cc] + 1)
			end
		end
		d[r, c] = best
	end
	@inbounds for r in nrows:-1:1, c in ncols:-1:1
		d[r, c] == 0 && continue
		best = d[r, c]
		for (dr, dc) in ((1, 1), (1, 0), (1, -1), (0, 1))
			rr, cc = r + dr, c + dc
			if 1 <= rr <= nrows && 1 <= cc <= ncols
				best = min(best, d[rr, cc] + 1)
			end
		end
		d[r, c] = best
	end
	[v >= far ? -1 : v for v in d]
end

"""
SciPy's `binary_erosion` of `mask` by the cross of the four neighbours, `iterations` times, the pixels beyond the
grid taken as set (`border_value=True`).
"""
function erode_cross(mask::AbstractMatrix{Bool}, iterations::Int)
	nrows, ncols = size(mask)
	at(m, r, c) = (1 <= r <= nrows && 1 <= c <= ncols) ? m[r, c] : true
	for _ in 1:iterations
		previous = mask
		mask = [previous[r, c] && at(previous, r - 1, c) && at(previous, r + 1, c) && at(previous, r, c - 1) &&
			at(previous, r, c + 1) for r in 1:nrows, c in 1:ncols]
	end
	mask
end

"""
scikit-image's `AffineTransform(rotation=angle, scale=(sx, 1))` as its matrix, (a0, a1, a2, b0, b1, b2), the
trigonometry the C library's, as Python's `math`.
"""
function rotation_matrix(angle::Float64, sx::Float64, tx::Float64 = 0.0, ty::Float64 = 0.0)
	c, s = libm_cos(angle), libm_sin(angle)
	(sx * (c + 0.0 * s), -1.0 * (0.0 * c + s), tx, sx * (s - 0.0 * c), -1.0 * (0.0 * s - c), ty)
end

"""The points (x, y) mapped by the matrix `m` (a0, a1, a2, b0, b1, b2), as scikit-image applies a transform."""
apply_matrix(m, points) = [(x * m[1] + y * m[2] + m[3], x * m[4] + y * m[5] + m[6]) for (x, y) in points]

"""The inverse of the matrix `m` (a0, a1, a2, b0, b1, b2), solved against the identity, as `numpy.linalg.inv` does."""
function invert_matrix(m)
	a = [m[1] m[2] m[3]; m[4] m[5] m[6]; 0.0 0.0 1.0]
	inverse = LinearAlgebra.LAPACK.gesv!(a, Matrix{Float64}(LinearAlgebra.I, 3, 3))[1]
	(inverse[1, 1], inverse[1, 2], inverse[1, 3], inverse[2, 1], inverse[2, 2], inverse[2, 3])
end

"""
Kraken's `_rotate` of the grid `patch` by `angle`, its width scaled by `scale`, on SciPy's path: the matrix and the
size of the rotated grid, which SciPy's `affine_transform` fills by the nearest pixel (order 0, mode "grid-constant"),
`cval` outside the patch, each input coordinate summed by fused multiply-adds.
"""
function rotate_patch(patch::Matrix{Float64}, angle::Float64, scale::Float64, cval::Float64)
	rows, cols = size(patch)
	m = rotation_matrix(angle, 1 / scale)
	corners = apply_matrix(invert_matrix(m), [(0.0, 0.0), (0.0, rows - 1.0), (cols - 1.0, rows - 1.0), (cols - 1.0, 0.0)])
	minc, minr = minimum(first, corners), minimum(last, corners)
	maxc, maxr = maximum(first, corners), maximum(last, corners)
	out_rows, out_cols = around(maxr - minr + 1), around(maxc - minc + 1)
	tx, ty = apply_matrix(m, [(minc, minr)])[1]
	m = rotation_matrix(angle, 1 / scale, tx, ty)
	out = Matrix{Float64}(undef, out_rows, out_cols)
	@inbounds for j in 0:out_cols - 1, i in 0:out_rows - 1
		# SciPy's matrix is the transform's with rows and columns swapped
		row = fma(Float64(j), m[4], fma(Float64(i), m[5], m[6]))
		col = fma(Float64(j), m[1], fma(Float64(i), m[2], m[3]))
		r = unsafe_trunc(Int, floor(row + 0.5))
		c = unsafe_trunc(Int, floor(col + 0.5))
		out[i + 1, j + 1] = (0 <= r < rows && 0 <= c < cols) ? patch[r + 1, c + 1] : cval
	end
	m, out
end

const MASK_VALUE = 99999.0

"""
Kraken's `_calc_seam`: the seam on one side of `baseline` (points (x, y)) through `polygon`, the region on that side:
the image features within the region, raised by the distance from the baseline, rotated so that the line runs level
and scaled to at most 600 pixels along it, then the path of least sum from its left to its right edge, each step up,
down or level, held within a standard deviation of its mean height and rotated back.
"""
function calc_seam(baseline::Vector{NTuple{2, Int}}, polygon::Vector{NTuple{2, Int}}, angle::Float64,
		features::Matrix{Float64}; bias::Float64 = 150.0)
	frows, fcols = size(features)
	c_min, c_max = minimum(first, polygon), maximum(first, polygon)
	r_min, r_max = minimum(last, polygon), maximum(last, polygon)
	patch = features[r_min + 1:min(r_max + 2, frows), c_min + 1:min(c_max + 2, fcols)]
	prows, pcols = size(patch)
	# the patch's distance from the baseline, in king's moves
	far = trues(prows, pcols)
	for k in 1:length(baseline) - 1
		(x0, y0), (x1, y1) = baseline[k], baseline[k + 1]
		rr, cc = draw_line(y0 - r_min, x0 - c_min, y1 - r_min, x1 - c_min)
		for (r, c) in zip(rr, cc)
			far[r + 1, c + 1] = false
		end
	end
	dist_bias = chessboard_distance(far)
	# the pixels outside the region, eroded twice
	inside = polygon_mask([(x - c_min, y - r_min) for (x, y) in polygon], pcols, prows)
	outside = erode_cross(permutedims(.!inside), 2)
	patch[outside] .= MASK_VALUE
	# NumPy's boolean index takes the pixels in row order
	kept = Float64[]
	for r in 1:prows, c in 1:pcols
		patch[r, c] != MASK_VALUE && push!(kept, patch[r, c])
	end
	weight = (numpy_sum(kept) / length(kept)) / bias
	patch .+= dist_bias .* weight
	extrema = [(baseline[1][1] - c_min, baseline[1][2] - r_min), (baseline[end][1] - c_min, baseline[end][2] - r_min)]
	extent = abs(libm_cos(angle)) >= abs(libm_sin(angle)) ? c_max - c_min : r_max - r_min
	extent == 0 && error("The region has no extent along the line.")
	scale = min(1.0, 600 / extent)
	m, rotated = rotate_patch(patch, angle, scale, MASK_VALUE)
	x_offsets = sort([around(x) for (x, _) in apply_matrix(invert_matrix(m), [(Float64(x), Float64(y)) for (x, y) in extrema])])
	x_offsets[1] < 0 && error("The line starts before the rotated patch.")
	rotated = rotated[:, x_offsets[1] + 1:min(x_offsets[2] + 1, size(rotated, 2))]
	padded = vcat(fill(Inf, 1, size(rotated, 2)), rotated, fill(Inf, 1, size(rotated, 2)))
	r, c = size(padded)
	c > 1 || error("The seam has no steps.")
	backtrack = zeros(Int, c - 1, r - 2)
	T = Vector{Float32}(undef, r - 2)
	@inbounds for i in 1:c - 1
		for k in 1:r - 2
			a, b, d = padded[k, i], padded[k + 1, i], padded[k + 2, i]
			# NumPy's min and argmin, the first of equal values
			best, arg = a, 0
			b < best && ((best, arg) = (b, 1))
			d < best && ((best, arg) = (d, 2))
			T[k] = Float32(best)
			backtrack[i, k] = arg + k - 2
		end
		for k in 1:r - 2
			padded[k + 1, i + 1] += Float64(T[k])
		end
	end
	j = argmin(padded[2:r - 1, c]) - 1
	seam = NTuple{2, Int}[]
	for i in c - 2:-1:-1
		push!(seam, (i + x_offsets[1] + 1, j))
		i >= 0 && (j = backtrack[i + 1, j + 1])
	end
	reverse!(seam)
	heights = [Float64(y) for (_, y) in seam]
	mean = numpy_sum(heights) / length(heights)
	std = sqrt(numpy_sum([(h - mean) * (h - mean) for h in heights]) / length(heights))
	seam = [(Float64(x), Float64(unsafe_trunc(Int, min(max(Float64(y), mean - std), mean + std)))) for (x, y) in seam]
	points = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in apply_matrix(m, seam)]
	filter!(p -> min(p[1], p[2]) >= 0 && p[1] < pcols && p[2] < prows && !outside[p[2] + 1, p[1] + 1], points)
	[(x + c_min, y + r_min) for (x, y) in points]
end

"""
Kraken's `_closest_intersects`: for each ray from `pts[i]` to `bounds[i]`, the point nearest to `pts[i]` where it
crosses `side`, looked for first among the parts of `side` near the first 100 pixels of the ray (by GEOS's STR tree),
then on the whole of `side`; `bounds[i]` where the ray crosses it nowhere.
"""
function closest_intersects(pts::Vector{NTuple{2, Float64}}, bounds::Vector{NTuple{2, Int}}, side; near::Float64 = 100.0)
	rays = [LibGEOS.LineString([[p[1], p[2]], [Float64(b[1]), Float64(b[2])]]) for (p, b) in zip(pts, bounds)]
	parts = geos_parts(side)
	tree = LibGEOS.STRtree(parts; nodecapacity = 10)
	index = IdDict{Any, Int}(part => k for (k, part) in enumerate(parts))
	out = [(Float64(b[1]), Float64(b[2])) for b in bounds]
	for (i, (p, ray)) in enumerate(zip(pts, rays))
		reach = LibGEOS.interpolate(ray, near)
		short = LibGEOS.LineString([[p[1], p[2]], [LibGEOS.getGeomX(reach), LibGEOS.getGeomY(reach)]])
		found = [index[g] for g in LibGEOS.query(tree, short)]
		inter = nothing
		if !isempty(found)
			nearby = LibGEOS.MultiLineString([geos_coordinates(parts[k]) for k in found])
			inter = LibGEOS.intersection(ray, nearby)
		end
		point = LibGEOS.Point(p[1], p[2])
		usable(g) = g isa Union{LibGEOS.Point, LibGEOS.MultiPoint, LibGEOS.GeometryCollection}
		if inter === nothing || !usable(inter) || LibGEOS.isEmpty(inter) || !(LibGEOS.distance(point, inter) <= near)
			inter = LibGEOS.intersection(ray, side)
		end
		LibGEOS.isEmpty(inter) && continue
		usable(inter) || error("No intersection with the boundaries.")
		nearest = LibGEOS.nearestPoints(point, inter)[2]
		out[i] = (LibGEOS.getGeomX(nearest), LibGEOS.getGeomY(nearest))
	end
	out
end

"""
Kraken's `_ray_intersect_boundaries`: where each ray from `rays[i]` in `direction` leaves the box from (0, 0) to
`box`.
"""
function ray_exits(rays::Vector{NTuple{2, Float64}}, direction::NTuple{2, Float64}, box::NTuple{2, Float64})
	map(rays) do p
		t = Inf
		for i in 1:2
			d = direction[i]
			d == 0 && continue
			t = min(t, ((d > 0 ? box[i] : 0.0) - p[i]) / d)
		end
		t = max(t, 0.0)
		(p[1] + direction[1] * t, p[2] + direction[2] * t)
	end
end

"""
Kraken's `_calc_roi`: the points of `line` every 10 pixels, and from each of them where the region above and below
the line ends, at the page's edge or at the nearest of `others` (lines and regions) on that side, each grown by a
pixel.
"""
function calc_roi(line::Vector{NTuple{2, Float64}}, bounds::NTuple{2, Float64}, others::Vector{Vector{NTuple{2, Int}}},
		p_dir::NTuple{2, Float64})
	ls = LibGEOS.LineString([[x, y] for (x, y) in line])
	len = LibGEOS.geomLength(ls)
	steps = max(0, unsafe_trunc(Int, ceil((len - 1 - 10) / 10)))
	middle = [LibGEOS.interpolate(ls, 10.0 + 10.0 * k) for k in 0:steps - 1]
	ip_line = vcat([line[1]], [(LibGEOS.getGeomX(p), LibGEOS.getGeomY(p)) for p in middle], [line[end]])
	box = (bounds[1] + 1, bounds[2] + 1)
	upper = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in ray_exits(ip_line, (p_dir[2], -p_dir[1]), box)]
	bottom = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in ray_exits(ip_line, (-p_dir[2], p_dir[1]), box)]
	upper_polygon = geos_polygon(vcat(ip_line, reverse(upper)))
	bottom_polygon = geos_polygon(vcat(ip_line, reverse(bottom)))
	adjacent = [LibGEOS.LineString([[Float64(x), Float64(y)] for (x, y) in o]) for o in others]
	in_upper = [LibGEOS.intersects(upper_polygon, a) for a in adjacent]
	in_bottom = [LibGEOS.intersects(bottom_polygon, a) && !u for (a, u) in zip(adjacent, in_upper)]
	edge(points) = LibGEOS.LineString([[Float64(x), Float64(y)] for (x, y) in points])
	side_a = LibGEOS.boundary(LibGEOS.buffer(union_all(vcat([edge(upper)], adjacent[in_upper])), 1.0, 16))
	side_b = LibGEOS.boundary(LibGEOS.buffer(union_all(vcat([edge(bottom)], adjacent[in_bottom])), 1.0, 16))
	env_up = closest_intersects(ip_line, upper, side_a)
	env_bottom = closest_intersects(ip_line, bottom, side_b)
	uint(points) = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in points]
	uint(env_up), uint(env_bottom), ip_line
end

"""
	line_outline(line, others, features; topline=false) -> Union{Vector{NTuple{2, Int}}, Nothing}

The outline of the baseline `line` (points (x, y)), as Kraken's `calculate_polygonal_environment` computes it for one
line: the region it may take, bounded by the page and by `others` (the other baselines and the regions that hold it),
cut on each side by the seam of least `features` (as `image_features` makes them), the seam on the side of the text
moved 8 pixels from the baseline (or less where the region is shallower), both seams simplified by 5 pixels and moved
4 pixels outwards. `topline` is false for baselines, true for toplines and `nothing` for centre lines. `nothing` where
Kraken finds no outline.
"""
function line_outline(line::Vector{NTuple{2, Int}}, others::Vector{Vector{NTuple{2, Int}}}, features::Matrix{Float64};
		topline::Union{Bool, Nothing} = false, rethrow_errors::Bool = false)
	try
		return calc_outline(line, others, features, topline)
	catch e
		rethrow_errors && rethrow()
		return nothing
	end
end

function calc_outline(line::Vector{NTuple{2, Int}}, others::Vector{Vector{NTuple{2, Int}}}, features::Matrix{Float64},
		topline::Union{Bool, Nothing})
	frows, fcols = size(features)
	bounds = (Float64(fcols) - 1, Float64(frows) - 1)
	end_points = (line[1], line[end])
	points = [(Float64(x), Float64(y)) for (x, y) in line]
	offset = topline === nothing ? 0 : 8
	n = length(points)
	dx = [points[k + 1][1] - points[k][1] for k in 1:n - 1]
	dy = [points[k + 1][2] - points[k][2] for k in 1:n - 1]
	lengths = [sqrt(a * a + b * b) for (a, b) in zip(dx, dy)]
	total = numpy_sum(lengths)
	px = strided_sum([a * l / total for (a, l) in zip(dx, lengths)]) / (n - 1)
	py = strided_sum([b * l / total for (b, l) in zip(dy, lengths)]) / (n - 1)
	norm = sqrt(px * px + py * py)
	p_dir = (px / norm, py / norm)
	env_up, env_bottom, ip_line = calc_roi(points, bounds, others, p_dir)
	env, normal = topline === true ? (env_bottom, (-p_dir[2], p_dir[1])) : (env_up, (p_dir[2], -p_dir[1]))
	offset_line = map(zip(env, ip_line)) do (e, p)
		ex, ey = Float64(e[1]) - p[1], Float64(e[2]) - p[2]
		depth = sqrt(ex * ex + ey * ey)
		k = max(min(Float64(offset), depth - 2), 0.0)
		(p[1] + k * normal[1], p[2] + k * normal[2])
	end
	offset_baseline = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in offset_line]
	extract_patch(env_up, env_bottom, line, offset_baseline, end_points, p_dir, topline, offset, features)
end

"""
Kraken's `_extract_patch`: the outline of the line from the seams above and below it, within the union of the regions
above and below.
"""
function extract_patch(env_up, env_bottom, baseline, offset_baseline, end_points, p_dir, topline, offset, features)
	upper_polygon = vcat(baseline, reverse(env_up))
	bottom_polygon = vcat(baseline, reverse(env_bottom))
	upper_offset_polygon = vcat(offset_baseline, reverse(env_up))
	bottom_offset_polygon = vcat(offset_baseline, reverse(env_bottom))
	angle = libm_atan2(p_dir[2], p_dir[1])
	roi_polygon = union_all([geos_polygon(upper_polygon), geos_polygon(bottom_polygon)])
	if topline === true
		upper_seam = calc_seam(baseline, upper_polygon, angle, features)
		bottom_seam = calc_seam(offset_baseline, bottom_offset_polygon, angle, features)
	else
		upper_seam = calc_seam(offset_baseline, upper_offset_polygon, angle, features)
		bottom_seam = calc_seam(baseline, bottom_polygon, angle, features)
	end
	seam_line(points) = LibGEOS.LineString([[Float64(x), Float64(y)] for (x, y) in points])
	upper = LibGEOS.topologyPreserveSimplify(seam_line(upper_seam), 5.0)
	bottom = LibGEOS.topologyPreserveSimplify(seam_line(bottom_seam), 5.0)
	shift = Float64(offset ÷ 2)
	ints(g) = [(unsafe_trunc(Int, x), unsafe_trunc(Int, y)) for (x, y) in geos_points(g)]
	moved = offset == 0 ? nothing : offset_curve(upper, -shift)
	upper_points = (offset == 0 || moved isa LibGEOS.MultiLineString) ? ints(upper) : reverse(ints(moved))
	moved = offset == 0 ? nothing : offset_curve(bottom, shift)
	bottom_points = (offset == 0 || moved isa LibGEOS.MultiLineString) ? ints(bottom) : ints(moved)
	(isempty(upper_points) || isempty(bottom_points)) && error("A seam is empty.")
	polygon = geos_polygon(vcat([end_points[1]], upper_points, [end_points[2]], reverse(bottom_points)))
	if !LibGEOS.isValid(polygon)
		polygon = geos_polygon(vcat([end_points[2]], upper_points, [end_points[1]], bottom_points))
	end
	LibGEOS.isValid(polygon) || error("Invalid bounding polygon computed.")
	outline = LibGEOS.boundary(LibGEOS.intersection(roi_polygon, polygon))
	outline isa Union{LibGEOS.LineString, LibGEOS.LinearRing} || error("The outline is not a single ring.")
	ints(outline)
end
