# The operations of Pillow 13 that Kraken 7.2 uses to prepare a line image, computed as Pillow's C code computes them
# (libImaging/Draw.c, Geometry.c, Resample.c, Convert.c, GetBBox.c), so that a line comes out with the same pixels.
# An image is an array of bytes (bands, width, height), pixel (x, y), counted from 0 as Pillow counts, at
# [:, x + 1, y + 1]; it has one band (grey) or three (RGB).

"""Pillow's `crop` of `image` to the box from (`x0`, `y0`) to (`x1`, `y1`), the end excluded, inside the image."""
crop(image::Array{UInt8, 3}, x0::Int, y0::Int, x1::Int, y1::Int) = image[:, x0 + 1:x1, y0 + 1:y1]

"""An edge of a polygon, as Draw.c's `Edge`."""
struct Edge
	x0::Int
	y0::Int
	xmin::Int
	ymin::Int
	xmax::Int
	ymax::Int
	dx::Float32
end

function Edge(x0::Int, y0::Int, x1::Int, y1::Int; xmin = min(x0, x1), xmax = max(x0, x1))
	dx = y0 == y1 ? 0.0f0 : Float32(x1 - x0) / Float32(y1 - y0)
	Edge(x0, y0, xmin, min(y0, y1), xmax, max(y0, y1), dx)
end

roundup(f::Float32) = f >= 0 ? Int(floor(Float64(f + 0.5f0))) : -Int(floor(Float64(abs(f) + 0.5f0)))
rounddown(f::Float32) = f >= 0 ? Int(ceil(Float64(f - 0.5f0))) : -Int(ceil(Float64(abs(f) - 0.5f0)))

"""
	polygon_mask(points, width, height) -> BitMatrix

The pixels (width, height) that Pillow's `ImageDraw.polygon` fills for the polygon `points` (integer x, y pairs), as
`ImagingDrawPolygon` and `polygon_generic` fill a polygon in an 8-bit image.
"""
function polygon_mask(points::AbstractVector{<:NTuple{2, Integer}}, width::Int, height::Int)
	mask = falses(width, height)
	hline(x0, y, x1) = if 0 <= y < height
		x0 = max(x0, 0)
		x1 = min(x1, width - 1)
		x0 <= x1 && (mask[x0 + 1:x1 + 1, y + 1] .= true)
	end
	count = length(points)
	edges = Edge[]
	for i in 1:count - 1
		(x0, y0), (x1, y1) = points[i], points[i + 1]
		if y0 == y1 && i != 1 && y0 == points[i - 1][2]
			# a horizontal edge right after another: the two are one edge
			previous = points[i - 1][1]
			if x1 > x0 && x0 > previous
				edges[end] = Edge(edges[end].x0, edges[end].y0, edges[end].xmin, edges[end].ymin, x1, edges[end].ymax, edges[end].dx)
				continue
			elseif x1 < x0 && x0 < previous
				edges[end] = Edge(edges[end].x0, edges[end].y0, x1, edges[end].ymin, edges[end].xmax, edges[end].ymax, edges[end].dx)
				continue
			end
		end
		push!(edges, Edge(x0, y0, x1, y1))
	end
	if points[end] != points[1]
		push!(edges, Edge(points[end]..., points[1]...))
	end
	ymin, ymax = height - 1, 0
	table = Edge[]
	for e in edges
		ymin = min(ymin, e.ymin)
		ymax = max(ymax, e.ymax)
		if e.ymin == e.ymax
			hline(e.xmin, e.ymin, e.xmax)
			continue
		end
		push!(table, e)
	end
	ymin = max(ymin, 0)
	ymax = min(ymax, height)
	# one rounding, as Pillow's wheels for Apple silicon compute (y - y0) * dx + x0: their compiler fuses it (an FMA)
	at(e, y) = fma(Float32(y - e.y0), e.dx, Float32(e.x0))
	for y in ymin:ymax
		xx = Float32[]
		for (i, current) in enumerate(table)
			current.ymin <= y <= current.ymax || continue
			push!(xx, at(current, y))
			if y == current.ymax && y < ymax
				# needed to draw consistent polygons
				push!(xx, xx[end])
			elseif (y == current.ymin || y == current.ymax) && current.dx != 0
				# connect discontiguous corners
				for k in 1:i - 1
					other = table[k]
					((y != other.ymin && y != other.ymax) || other.dx == 0) && continue
					if round(xx[end], RoundNearestTiesAway) == round(at(other, y), RoundNearestTiesAway)
						offset = y == current.ymax ? -1 : 1
						adjacent = at(current, y + offset)
						if other.ymin <= y + offset <= other.ymax
							adjacent_other = at(other, y + offset)
							if xx[end] > adjacent + 1 && xx[end] > adjacent_other + 1
								xx[end] = round(Float32(max(Float64(adjacent), Float64(adjacent_other))), RoundNearestTiesAway) + 1
							elseif xx[end] < adjacent - 1 && xx[end] < adjacent_other - 1
								xx[end] = round(Float32(min(Float64(adjacent), Float64(adjacent_other))), RoundNearestTiesAway) - 1
							end
							break
						end
					end
				end
			end
		end
		sort!(xx)
		for i in 2:2:length(xx)
			hline(roundup(xx[i - 1]), y, rounddown(xx[i]))
		end
	end
	mask
end

"""Pillow's paste of `image` onto a black image through `mask`, which keeps the pixels inside the mask."""
function apply_mask(image::Array{UInt8, 3}, mask::BitMatrix)
	out = copy(image)
	for y in axes(image, 3), x in axes(image, 2)
		mask[x, y] || (out[:, x, y] .= 0x00)
	end
	out
end

"""
The value at (`xin`, `yin`) of band `band` of `image` by Pillow's bilinear filter (`bilinear_filter8` and
`bilinear_filter32RGB`), `nothing` outside the image.
"""
@inline function bilinear(image::Array{UInt8, 3}, band::Int, xin::Float64, yin::Float64)
	xsize, ysize = size(image, 2), size(image, 3)
	(xin < 0 || xin >= xsize || yin < 0 || yin >= ysize) && return nothing
	xin -= 0.5
	yin -= 0.5
	x = xin < 0 ? Int(floor(xin)) : unsafe_trunc(Int, xin)
	y = yin < 0 ? Int(floor(yin)) : unsafe_trunc(Int, yin)
	dx, dy = xin - x, yin - y
	xclip(v) = v < 0 ? 0 : v < xsize ? v : xsize - 1
	yclip(v) = v < 0 ? 0 : v < ysize ? v : ysize - 1
	x0, x1 = xclip(x) + 1, xclip(x + 1) + 1
	row = yclip(y) + 1
	a, b = Float64(image[band, x0, row]), Float64(image[band, x1, row])
	# a + (b - a) * d with one rounding, as the C of Pillow's wheels for Apple silicon fuses it (an FMA)
	v1 = fma(b - a, dx, a)
	if 0 <= y + 1 < ysize
		a, b = Float64(image[band, x0, y + 2]), Float64(image[band, x1, y + 2])
		v2 = fma(b - a, dx, a)
	else
		v2 = v1
	end
	fma(v2 - v1, dy, v1)
end

"""
The pixels of `out` in the box from (`x0`, `y0`) to (`x1`, `y1`) of the output, each read from `image` at the point
`map(x + 0.5, y + 0.5)` (x and y counted from the box's corner) by the bilinear filter, as Pillow's
`ImagingGenericTransform`; a pixel whose point falls outside `image` is set to 0.
"""
function generic_transform!(out::Array{UInt8, 3}, image::Array{UInt8, 3}, box::NTuple{4, Int}, map::F) where {F}
	x0, y0 = max(box[1], 0), max(box[2], 0)
	x1, y1 = min(box[3], size(out, 2)), min(box[4], size(out, 3))
	for y in y0:y1 - 1, x in x0:x1 - 1
		# Pillow counts from the box's corner after clipping it to the output
		xin, yin = map(x - x0 + 0.5, y - y0 + 0.5)
		for band in axes(image, 1)
			v = bilinear(image, band, xin, yin)
			# Pillow casts the double to an unsigned byte, which truncates
			out[band, x + 1, y + 1] = v === nothing ? 0x00 : unsafe_trunc(UInt8, v)
		end
	end
	out
end

"""
Pillow's `transform` of `image` to an image of `width` by `height` by the affine map `a` (six numbers), bilinear. The
map is fused as Pillow's `affine_transform` is compiled for Apple silicon: `fma(a0, x, a1 * y) + a2`.
"""
function affine(image::Array{UInt8, 3}, width::Int, height::Int, a::NTuple{6, Float64})
	out = zeros(UInt8, size(image, 1), width, height)
	generic_transform!(out, image, (0, 0, width, height), (x, y) -> (fma(a[1], x, a[2] * y) + a[3], fma(a[4], x, a[5] * y) + a[6]))
end

"""
Pillow's `transform` of `image` by a mesh, bilinear: each element of `mesh` is an output box (x0, y0, x1, y1) and the
quadrilateral of the source it is read from (NW, SW, SE and NE corners, eight numbers). The map is fused as Pillow's
`quad_transform` is compiled for Apple silicon, one FMA for each product added.
"""
function mesh_transform(image::Array{UInt8, 3}, width::Int, height::Int, mesh)
	out = zeros(UInt8, size(image, 1), width, height)
	for (box, quad) in mesh
		w, h = box[3] - box[1], box[4] - box[2]
		nw, sw, se, ne = quad[1:2], quad[3:4], quad[5:6], quad[7:8]
		x0, y0 = Float64(nw[1]), Float64(nw[2])
		as, at = 1.0 / w, 1.0 / h
		a = (x0, (ne[1] - x0) * as, (sw[1] - x0) * at, (se[1] - sw[1] - ne[1] + x0) * as * at,
			y0, (ne[2] - y0) * as, (sw[2] - y0) * at, (se[2] - sw[2] - ne[2] + y0) * as * at)
		generic_transform!(out, image, Tuple(Int.(box)),
			(x, y) -> (fma(a[4] * x, y, fma(a[3], y, fma(a[2], x, a[1]))), fma(a[8] * x, y, fma(a[7], y, fma(a[6], x, a[5])))))
	end
	out
end

"""Pillow's `getbbox`: the box (x0, y0, x1, y1) of the pixels with a band not 0, `nothing` when there is none."""
function bounding_box(image::Array{UInt8, 3})
	nonzero = dropdims(any(!iszero, image; dims = 1); dims = 1)
	columns = findall(vec(any(nonzero; dims = 2)))
	rows = findall(vec(any(nonzero; dims = 1)))
	isempty(rows) && return nothing
	(first(columns) - 1, first(rows) - 1, last(columns), last(rows))
end

"""Pillow's `convert("L")`: the grey of each pixel (width, height), ITU-R 601-2 in fixed point for an RGB image."""
function grey(image::Array{UInt8, 3})
	size(image, 1) == 1 && return image[1, :, :]
	map(CartesianIndices((size(image, 2), size(image, 3)))) do i
		r, g, b = Int(image[1, i]), Int(image[2, i]), Int(image[3, i])
		UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)
	end
end

"""Pillow's Lanczos filter: the sinc truncated at 3."""
function lanczos(x::Float64)
	sinc(v) = v == 0.0 ? 1.0 : sin(v * π) / (v * π)
	-3.0 <= x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0
end

const PRECISION_BITS = 32 - 8 - 2

"""
The bounds (first input, number of inputs) and the fixed-point coefficients of each of `outsize` outputs resampled
from `insize` inputs by Pillow's Lanczos filter (`precompute_coeffs` and `normalize_coeffs_8bpc`).
"""
function coefficients(insize::Int, outsize::Int)
	scale = Float64(Float32(insize)) / outsize
	filterscale = max(scale, 1.0)
	support = 3.0 * filterscale
	ksize = Int(ceil(support)) * 2 + 1
	bounds = Vector{NTuple{2, Int}}(undef, outsize)
	kk = zeros(Int32, ksize, outsize)
	inverse = 1.0 / filterscale
	for xx in 0:outsize - 1
		center = 0.0 + (xx + 0.5) * scale
		xmin = max(unsafe_trunc(Int, center - support + 0.5), 0)
		xmax = min(unsafe_trunc(Int, center + support + 0.5), insize) - xmin
		k = [lanczos((x + xmin - center + 0.5) * inverse) for x in 0:xmax - 1]
		total = sum(k; init = 0.0)
		total != 0.0 && (k ./= total)
		for x in 1:xmax
			kk[x, xx + 1] = k[x] < 0 ? unsafe_trunc(Int32, -0.5 + k[x] * (1 << PRECISION_BITS)) :
				unsafe_trunc(Int32, 0.5 + k[x] * (1 << PRECISION_BITS))
		end
		bounds[xx + 1] = (xmin, xmax)
	end
	bounds, kk
end

clip8(s::Int32) = UInt8(clamp(s >> PRECISION_BITS, 0, 255))

"""
	resize(grey, width, height) -> Matrix{UInt8}

Pillow's `resize` of a grey image (width, height) by the Lanczos filter, with its passes in Pillow's order and its
rounding to bytes between them (`ImagingResampleInner`).
"""
function resize(image::Matrix{UInt8}, width::Int, height::Int)
	inwidth, inheight = size(image)
	(inwidth, inheight) == (width, height) && return copy(image)
	horizontal(img) = begin
		bounds, kk = coefficients(size(img, 1), width)
		out = Matrix{UInt8}(undef, width, size(img, 2))
		for y in axes(img, 2), xx in 1:width
			xmin, xmax = bounds[xx]
			s = Int32(1) << (PRECISION_BITS - 1)
			for x in 1:xmax
				s += Int32(img[x + xmin, y]) * kk[x, xx]
			end
			out[xx, y] = clip8(s)
		end
		out
	end
	vertical(img, rows) = begin
		bounds, kk = rows
		out = Matrix{UInt8}(undef, size(img, 1), height)
		for yy in 1:height, x in axes(img, 1)
			ymin, ymax = bounds[yy]
			s = Int32(1) << (PRECISION_BITS - 1)
			for y in 1:ymax
				s += Int32(img[x, y + ymin]) * kk[y, yy]
			end
			out[x, yy] = clip8(s)
		end
		out
	end
	need_horizontal, need_vertical = width != inwidth, height != inheight
	horizontal_first = !((inheight - height) > 0 && (inheight - height) > (inwidth - width) * 2)
	rows = coefficients(inheight, height)
	if horizontal_first
		img = image
		if need_horizontal
			# the horizontal pass computes only the rows the vertical pass reads
			first_row = rows[1][1][1]
			last_row = rows[1][end][1] + rows[1][end][2]
			img = horizontal(image[:, first_row + 1:last_row])
			rows = ([(b[1] - first_row, b[2]) for b in rows[1]], rows[2])
		end
		need_vertical ? vertical(img, rows) : img
	else
		img = need_vertical ? vertical(image, rows) : image
		need_horizontal ? horizontal(img) : img
	end
end
