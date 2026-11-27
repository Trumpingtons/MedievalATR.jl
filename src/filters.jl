# The image filters of SciPy 1.16 and scikit-image 0.26 that Kraken 7.2's segmenter uses, computed as their C code
# computes them (scipy/ndimage/src/ni_filters.c), so that a map comes out with the same values. A map is a matrix
# (width, height), pixel (x, y) at [x + 1, y + 1]: NumPy's axis 0 (rows) is its second dimension, and axis 1 its first.

"""
NumPy's sum of `v` (`np.add.reduce` on a contiguous array of doubles): pairwise, with eight partial sums in blocks of
up to 128 values, as `pairwise_sum` in numpy/_core/src/umath/loops_utils.h.
"""
function numpy_sum(v::AbstractVector{Float64})
	n = length(v)
	n < 8 && return foldl(+, v; init = -0.0)
	if n <= 128
		r = [v[k] for k in 1:8]
		for i in 9:8:n - (n % 8)
			for k in 1:8
				r[k] += v[i + k - 1]
			end
		end
		total = ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
		for i in n - (n % 8) + 1:n
			total += v[i]
		end
		return total
	end
	half = n ÷ 2
	half -= half % 8
	numpy_sum(view(v, 1:half)) + numpy_sum(view(v, half + 1:n))
end

"""The C library's `exp`, which NumPy calls, and which Julia's own `exp` does not always match in the last bit."""
libm_exp(x::Float64) = ccall(:exp, Float64, (Float64,), x)

"""The C library's `sin`, `cos` and `atan2`, which NumPy's scalars call; Julia's own differ from them in the last bit."""
libm_sin(x::Float64) = ccall(:sin, Float64, (Float64,), x)
libm_cos(x::Float64) = ccall(:cos, Float64, (Float64,), x)
libm_atan2(y::Float64, x::Float64) = ccall(:atan2, Float64, (Float64, Float64), y, x)

"""
SciPy's `_gaussian_kernel1d`: the Gaussian of standard deviation `sigma` (order 0) or its first derivative (order 1),
sampled from -`radius` to `radius`.
"""
function gaussian_kernel(sigma::Float64, order::Int, radius::Int)
	order in (0, 1) || error("Gaussian kernels of order $(order) are not computed.")
	sigma2 = sigma * sigma
	x = -radius:radius
	phi = [libm_exp(-0.5 / sigma2 * Float64(k^2)) for k in x]
	phi ./= numpy_sum(phi)
	order == 0 && return phi
	# q(x) = -x / sigma^2, as the matrix operator of SciPy computes it
	q = -1 / sigma2
	[(0.0 + Float64(k) * q) * p for (k, p) in zip(x, phi)]
end

"""
SciPy's `correlate1d` of `input` with `weights` along dimension `dim`, in double precision and stored as `T`, the line
extended by `cval` (`mode` :constant) or by its mirror image (:reflect, d c b a | a b c d | d c b a), its sums as
NI_Correlate1D makes them: in pairs from the outside in for a symmetric or antisymmetric kernel, each step a fused
multiply-add as clang compiles it for arm64.
"""
function correlate1d(input::Matrix{<:AbstractFloat}, weights::Vector{Float64}, dim::Int; mode::Symbol = :constant,
		cval::Float64 = 0.0, T::Type{<:AbstractFloat} = eltype(input))
	mode in (:constant, :reflect) || error("The mode is :constant or :reflect.")
	n = length(weights)
	size1 = n ÷ 2
	size2 = n - size1 - 1
	symmetric = 0
	if isodd(n)
		symmetric = all(abs(weights[size1 + 1 + i] - weights[size1 + 1 - i]) <= eps(Float64) for i in 1:size1) ? 1 : 0
		if symmetric == 0
			symmetric = all(abs(weights[size1 + 1 + i] + weights[size1 + 1 - i]) <= eps(Float64) for i in 1:size1) ? -1 : 0
		end
	end
	fw(j) = weights[size1 + 1 + j]
	x = dim == 1 ? input : permutedims(input)
	length_, lines = size(x)
	mode == :reflect && length_ < max(size1, size2) && error("The line is shorter than the kernel's half.")
	out = Matrix{T}(undef, length_, lines)
	Threads.@threads for line in 1:lines
		buffer = fill(cval, size1 + length_ + size2)
		@inbounds for i in 1:length_
			buffer[size1 + i] = Float64(x[i, line])
		end
		if mode == :reflect
			@inbounds for k in 1:size1
				buffer[size1 + 1 - k] = Float64(x[k, line])
			end
			@inbounds for k in 1:size2
				buffer[size1 + length_ + k] = Float64(x[length_ + 1 - k, line])
			end
		end
		@inbounds for l in 1:length_
			c = size1 + l
			if symmetric > 0
				s = buffer[c] * fw(0)
				for j in -size1:-1
					s = fma(buffer[c + j] + buffer[c - j], fw(j), s)
				end
			elseif symmetric < 0
				s = buffer[c] * fw(0)
				for j in -size1:-1
					s = fma(buffer[c + j] - buffer[c - j], fw(j), s)
				end
			else
				s = buffer[c + size2] * fw(size2)
				for j in -size1:size2 - 1
					s = fma(buffer[c + j], fw(j), s)
				end
			end
			out[l, line] = T(s)
		end
	end
	dim == 1 ? out : permutedims(out)
end

"""
SciPy's `gaussian_filter` of `x` with standard deviation `sd` and the derivative orders `orders` (NumPy's axis 0, the
rows, then axis 1), mode "constant", the kernels truncated at `truncate` standard deviations.
"""
function gaussian_filter(x::Matrix{Float32}, sd::Float64, orders::NTuple{2, Int}; truncate::Float64 = 4.0)
	radius = unsafe_trunc(Int, truncate * sd + 0.5)
	# SciPy reverses the kernel, since it correlates
	y = correlate1d(x, reverse(gaussian_kernel(sd, orders[1], radius)), 2)
	correlate1d(y, reverse(gaussian_kernel(sd, orders[2], radius)), 1)
end

"""
Kraken's `_sato`: the Sato tubeness of `im` (scikit-image's, truncated at 8 standard deviations), the largest over the
scales 1, 3, 5, 7 and 9 of the largest eigenvalue of the Hessian of the inverted map, scaled by the square of the
scale, in single precision as NumPy computes it.
"""
function sato(im::Matrix{Float32})
	x = -im
	scales = 1:2:9
	results = Vector{Matrix{Float32}}(undef, length(scales))
	for (k, sigma) in enumerate(scales)
		sd = sigma / sqrt(2)
		gr = gaussian_filter(x, sd, (1, 0); truncate = 8.0)
		gc = gaussian_filter(x, sd, (0, 1); truncate = 8.0)
		hrr = gaussian_filter(gr, sd, (1, 0); truncate = 8.0)
		hrc = gaussian_filter(gr, sd, (0, 1); truncate = 8.0)
		hcc = gaussian_filter(gc, sd, (0, 1); truncate = 8.0)
		s2 = Float32(sigma^2)
		results[k] = map(hrr, hrc, hcc) do rr, rc, cc
			largest = (rr + cc) / 2.0f0 + sqrt(rc * rc + ((rr - cc) / 2.0f0) * ((rr - cc) / 2.0f0))
			s2 * max(largest, 0.0f0)
		end
	end
	reduce((a, b) -> max.(a, b), results)
end

"""
The image features on which Kraken cuts the outlines of the lines, `gaussian_filter(sobel(scaled), 0.5)`: the Sobel
magnitude of the grey page `scaled` (width, height) as scikit-image computes it, the page from 0 to 1, each direction a
3 by 3 correlation of SciPy (mode "reflect") and the magnitude the root of half the sum of their squares, then SciPy's
Gaussian of standard deviation 0.5 (mode "reflect"), all in double precision. The result is a grid (rows, columns).
"""
function image_features(scaled::Matrix{UInt8})
	image = permutedims(Float64.(scaled) .* (1.0 / 255))
	nrows, ncols = size(image)
	at(r, c) = image[clamp_reflect(r, nrows), clamp_reflect(c, ncols)]
	sobel = Matrix{Float64}(undef, nrows, ncols)
	Threads.@threads for c in 1:ncols
		@inbounds for r in 1:nrows
			# the kernels of ndimage.convolve, flipped, their non-zero weights in row order
			a = 0.0
			a = fma(-0.25, at(r - 1, c - 1), a)
			a = fma(-0.5, at(r - 1, c), a)
			a = fma(-0.25, at(r - 1, c + 1), a)
			a = fma(0.25, at(r + 1, c - 1), a)
			a = fma(0.5, at(r + 1, c), a)
			a = fma(0.25, at(r + 1, c + 1), a)
			b = 0.0
			b = fma(-0.25, at(r - 1, c - 1), b)
			b = fma(0.25, at(r - 1, c + 1), b)
			b = fma(-0.5, at(r, c - 1), b)
			b = fma(0.5, at(r, c + 1), b)
			b = fma(-0.25, at(r + 1, c - 1), b)
			b = fma(0.25, at(r + 1, c + 1), b)
			sobel[r, c] = sqrt(a * a + b * b) / sqrt(2.0)
		end
	end
	sd = 0.5
	radius = unsafe_trunc(Int, 4.0 * sd + 0.5)
	kernel = reverse(gaussian_kernel(sd, 0, radius))
	y = correlate1d(sobel, kernel, 1; mode = :reflect)
	correlate1d(y, kernel, 2; mode = :reflect)
end

"""The index `i` of a line of `n` values extended by its mirror image (SciPy's "reflect"), for one step past an end."""
clamp_reflect(i::Int, n::Int) = i < 1 ? 1 - i : i > n ? 2n + 1 - i : i
