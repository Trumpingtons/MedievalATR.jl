# The line images of a page, cut out along their baselines and boundary polygons as Kraken 7.2's `extract_polygons`
# cuts them (kraken/lib/segmentation.py), then prepared for the network as its `ImageInputTransforms` prepare them.

"""
	cut_line(image, baseline, boundary) -> Array{UInt8, 3}

The image of the line with `baseline` and `boundary` (vectors of integer x, y points on the page) cut out of `image`
(bands, width, height), straightened along its baseline, as Kraken cuts it: the pixels outside the boundary are black,
a line with a straight baseline (two points) is rotated, another is warped piecewise along its baseline, and the result
is cropped to its pixels that are not black. An error when the line lies outside the image or its baseline is shorter
than 5 pixels.
"""
function cut_line(image::Array{UInt8, 3}, baseline::AbstractVector, boundary::AbstractVector)
	pl = [Float64.((p[1], p[2])) for p in boundary]
	bl = [Float64.((p[1], p[2])) for p in baseline]
	length(bl) >= 2 && sum(hypot((bl[i + 1] .- bl[i])...) for i in 1:length(bl) - 1) >= 5 ||
		throw(ArgumentError("Baseline length below minimum 5px"))
	c_min, c_max = trunc(Int, minimum(first, pl)), trunc(Int, maximum(first, pl))
	r_min, r_max = trunc(Int, minimum(last, pl)), trunc(Int, maximum(last, pl))
	width, height = size(image, 2), size(image, 3)
	outside(points) = any(p -> p[1] < 0 || p[2] < 0, points) || maximum(first, points) >= width || maximum(last, points) >= height
	outside(pl) && throw(ArgumentError("Line polygon outside of image bounds"))
	outside(bl) && throw(ArgumentError("Baseline outside of image bounds"))
	patch = crop(image, c_min, r_min, c_max + 1, r_max + 1)
	if length(bl) == 2
		# Kraken's direction: the segment weighted by its share of the length, (d * L) / L, then made a unit vector
		dx, dy = bl[2][1] - bl[1][1], bl[2][2] - bl[1][2]
		len = sqrt(dx * dx + dy * dy)
		dx, dy = (dx * len) / len, (dy * len) / len
		norm = sqrt(dx * dx + dy * dy)
		angle = atan(dy / norm, dx / norm)
		offset = [(trunc(Int, p[1] - c_min), trunc(Int, p[2] - r_min)) for p in pl]
		patch = apply_mask(patch, polygon_mask(offset, size(patch, 2), size(patch, 3)))
		out = rotate(patch, angle)
	else
		out = warp(patch, pl, bl, c_min, r_min)
	end
	box = bounding_box(out)
	box === nothing ? out : crop(out, box...)
end

"""
The image rotated by `angle` (radians) about its corner and fitted to a new image, as Kraken's `_rotate` does it with
scikit-image's `AffineTransform` and Pillow's bilinear affine transform.
"""
function rotate(patch::Array{UInt8, 3}, angle::Float64)
	rows, cols = size(patch, 3), size(patch, 2)
	c, s = cos(angle), sin(angle)
	params = [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
	# scikit-image inverts the matrix with numpy.linalg.inv, a LAPACK solve against the identity
	inverse = LinearAlgebra.LAPACK.gesv!(copy(params), Matrix{Float64}(LinearAlgebra.I, 3, 3))[1]
	corners = [0.0 0.0; 0.0 rows - 1; cols - 1 rows - 1; cols - 1 0.0]
	mapped = [corners ones(4)] * permutedims(inverse)
	mapped = mapped[:, 1:2] ./ mapped[:, 3]
	minc, minr = minimum(mapped[:, 1]), minimum(mapped[:, 2])
	maxc, maxr = maximum(mapped[:, 1]), maximum(mapped[:, 2])
	out_rows, out_cols = round(Int, maxr - minr + 1), round(Int, maxc - minc + 1)
	translation = [minc minr 1.0] * permutedims(params)
	tx, ty = translation[1] / translation[3], translation[2] / translation[3]
	affine(patch, out_cols, out_rows, (c, -s, tx, s, c, ty))
end

"""
scikit-image's `approximate_polygon` (Douglas-Peucker) of `coords` (x, y points) with `tolerance`.
"""
function approximate_polygon(coords::Vector{NTuple{2, Float64}}, tolerance::Float64)
	n = length(coords)
	chain = falses(n)
	chain[1] = chain[n] = true
	dists = zeros(n)
	stack = [(1, n)]
	while !isempty(stack)
		start, stop = pop!(stack)
		r0, c0 = coords[start]
		r1, c1 = coords[stop]
		dr, dc = r1 - r0, c1 - c0
		segment_angle = -atan(dr, dc)
		segment_dist = c0 * sin(segment_angle) + r0 * cos(segment_angle)
		for k in start + 1:stop - 1
			r, c = coords[k]
			dr0, dc0, dr1, dc1 = r - r0, c - c0, r - r1, c - c1
			if dr0 * dr + dc0 * dc > 0 && -dr1 * dr - dc1 * dc > 0
				dists[k] = abs(r * cos(segment_angle) + c * sin(segment_angle) - segment_dist)
			else
				dists[k] = min(sqrt(dc0^2 + dr0^2), sqrt(dc1^2 + dr1^2))
			end
		end
		if stop - start > 1 && any(dists[k] > tolerance for k in start + 1:stop - 1)
			new_end = start + argmax(dists[start + 1:stop - 1])
			push!(stack, (new_end, stop))
			push!(stack, (start, new_end))
			chain[new_end] = true
		end
	end
	coords[chain]
end

"""scikit-image's `subdivide_polygon` of degree 2 (Chaikin's corner cutting), the ends kept for an open polygon."""
function subdivide_polygon(coords::Vector{NTuple{2, Float64}})
	circular = coords[1] == coords[end]
	out = NTuple{2, Float64}[]
	if circular
		points = coords[1:end - 1]
		n = length(points)
		for k in 1:n
			previous = points[k == 1 ? n : k - 1]
			push!(out, 0.25 .* points[k] .+ 0.75 .* previous, 0.75 .* points[k] .+ 0.25 .* previous)
		end
		push!(out, out[1])
	else
		for k in 1:length(coords) - 1
			push!(out, 0.75 .* coords[k] .+ 0.25 .* coords[k + 1], 0.25 .* coords[k] .+ 0.75 .* coords[k + 1])
		end
		out = [coords[1]; out; coords[end]]
	end
	out
end

"""
The line warped piecewise along its baseline `bl` (points of the page) so that the baseline becomes straight, as the
slow path of Kraken's `extract_polygons`: the boundary `pl` is subdivided, its points placed on the straightened
baseline, and the patch (cropped at `c_min`, `r_min`) mapped by Pillow's mesh transform.
"""
function warp(patch::Array{UInt8, 3}, pl::Vector{NTuple{2, Float64}}, bl::Vector{NTuple{2, Float64}}, c_min::Int, r_min::Int)
	length(pl) > 50 && (pl = approximate_polygon(pl, 2.0))
	full = subdivide_polygon(pl)
	segments = [bl[k + 1] .- bl[k] for k in 1:length(bl) - 1]
	norms = [sqrt(s[1]^2 + s[2]^2) for s in segments]
	normed = [segments[k] ./ norms[k] for k in eachindex(segments)]
	cum = cumsum([0.0; norms])
	bl_dst = [bl[1] .+ (cum[k], 0.0) for k in eachindex(cum)]
	pol_dst = map(full) do p
		diffs = [p .- bl[k] for k in eachindex(segments)]
		xs = [diffs[k][1] * normed[k][1] + diffs[k][2] * normed[k][2] for k in eachindex(segments)]
		segdist = [max(-xs[k], xs[k] - norms[k]) for k in eachindex(segments)]
		k = argmin(segdist)
		y = normed[k][1] * diffs[k][2] - normed[k][2] * diffs[k][1]
		(cum[k] + xs[k] + bl[1][1], y + bl[1][2])
	end
	c_dst_min, c_dst_max = trunc(Int, minimum(first, pol_dst)), trunc(Int, maximum(first, pol_dst))
	r_dst_min, r_dst_max = trunc(Int, minimum(last, pol_dst)), trunc(Int, maximum(last, pol_dst))
	out_rows, out_cols = r_dst_max - r_dst_min + 1, c_dst_max - c_dst_min + 1
	offset_polygon = [(trunc(Int, p[1] - c_min), trunc(Int, p[2] - r_min)) for p in full]
	offset_baseline = [p .- (c_min, r_min) for p in bl]
	offset_start = bl_dst[1] .- (c_dst_min, r_dst_min)
	patch = apply_mask(patch, polygon_mask(offset_polygon, size(patch, 2), size(patch, 3)))
	source, target = warping_envelope(offset_baseline, offset_start, (out_rows, out_cols))
	mesh = [((target[i]..., target[i + 3]...), (source[i]..., source[i + 1]..., source[i + 3]..., source[i + 2]...))
		for i in 1:2:length(source) - 3]
	mesh_transform(patch, out_cols, out_rows, mesh)
end

"""Kraken's `_bevelled_warping_envelope`: the source and target points of the mesh along a bent baseline."""
function warping_envelope(baseline, start, shape)
	as_int(p) = (trunc(Int, p[1]), trunc(Int, p[2]))
	dy = (-start[2], shape[1] - start[2])
	segments = [baseline[k + 1] .- baseline[k] for k in 1:length(baseline) - 1]
	lengths = [sqrt(s[1]^2 + s[2]^2) for s in segments]
	normed = [segments[k] ./ lengths[k] for k in eachindex(segments)]
	cum = cumsum([0.0; lengths])
	normals = [(-n[2], n[1]) for n in normed]
	ini = baseline[1] .- normed[1] .* start[1]
	source = Any[as_int(ini .+ dy[1] .* normals[1]), as_int(ini .+ dy[2] .* normals[1])]
	target = Any[(0, 0), (0, shape[1])]
	max_bevel = shape[1] / 3
	step = max_bevel / 2
	for k in 1:length(baseline) - 2
		pt = baseline[k + 1]
		previous, next = baseline[k] .- pt, baseline[k + 2] .- pt
		bevel_previous = previous ./ max(2.0, sqrt(previous[1]^2 + previous[2]^2) / max_bevel)
		bevel_next = next ./ max(2.0, sqrt(next[1]^2 + next[2]^2) / max_bevel)
		l_previous = sqrt(bevel_previous[1]^2 + bevel_previous[2]^2)
		l_next = sqrt(bevel_next[1]^2 + bevel_next[2]^2)
		nsteps = max(1.0, round((l_previous + l_next) / step))
		for i in 0:Int(nsteps)
			t = i / nsteps
			tpt = pt .+ (1 - t)^2 .* bevel_previous .+ t^2 .* bevel_next
			tx = start[1] + cum[k + 1] - (1 - t)^2 * l_previous + t^2 * l_next
			tnormal = (1 - t) .* normals[k] .+ t .* normals[k + 1]
			tnormal = tnormal ./ sqrt(tnormal[1]^2 + tnormal[2]^2)
			points = (as_int(tpt .+ dy[1] .* tnormal), as_int(tpt .+ dy[2] .* tnormal))
			targets = ((trunc(Int, tx), 0), (trunc(Int, tx), shape[1]))
			# avoid duplicate points, which make the mesh singular
			(points[1] == source[end - 1] || points[2] == source[end] || targets[1] == target[end - 1]) && continue
			push!(source, points...)
			push!(target, targets...)
		end
	end
	stop = baseline[end] .+ normed[end] .* (shape[2] - cum[end] - start[1])
	push!(source, stop .+ dy[1] .* normals[end], stop .+ dy[2] .* normals[end])
	push!(target, (shape[2], 0), (shape[2], shape[1]))
	source, target
end

"""
	prepare_line(line, height; pad = 16) -> Matrix{Float32}

The network's input (height, width) for the cut line image `line` (bands, width, height), as Kraken's
`ImageInputTransforms` make it: grey, scaled to `height` pixels high by Pillow's Lanczos filter, padded with `pad`
white columns on each side, scaled to 0 to 1 and inverted, so that ink is high.
"""
function prepare_line(line::Array{UInt8, 3}, height::Int; pad::Int = 16)
	g = grey(line)
	w, h = size(g)
	g = resize(g, trunc(Int, w * height / h), height)
	padded = fill(0xff, size(g, 1) + 2pad, height)
	padded[pad + 1:pad + size(g, 1), :] = g
	x = permutedims(Float32.(padded) .* Float32(1 / 255))
	maximum(x) .- x
end
