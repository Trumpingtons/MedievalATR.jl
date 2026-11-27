# MedievalATR's tests, against values computed by Pillow 13 and Kraken 7.2 with CATMuS Medieval (test/data, made by
# fixtures.py from the Kraken reference): polygon masks, Lanczos scaling, ALTO and PAGE reading, and the lines of a small
# drawn page as Kraken cuts, prepares and reads them. The last needs the model, downloaded from Zenodo the first time.
using MedievalATR, Test, JSON

const DATA = joinpath(@__DIR__, "data")
const FIXTURES = JSON.parsefile(joinpath(DATA, "fixtures.json"))

points(v) = v === nothing ? nothing : [(Int(p[1]), Int(p[2])) for p in v]

"""The filled runs [y, first x, last x] of a mask (width, height), counted from 0."""
function runs(mask::BitMatrix)
	out = Vector{Int}[]
	for y in axes(mask, 2)
		x = 1
		while x <= size(mask, 1)
			if mask[x, y]
				start = x
				while x < size(mask, 1) && mask[x + 1, y]
					x += 1
				end
				push!(out, [y - 1, start - 1, x - 1])
			end
			x += 1
		end
	end
	out
end

@testset "MedievalATR" begin
	@testset "polygon masks as Pillow draws them" begin
		got = [runs(MedievalATR.polygon_mask(points(c["points"]), c["width"], c["height"])) for c in FIXTURES["masks"]]
		@test got == [Vector{Vector{Int}}(c["runs"]) for c in FIXTURES["masks"]]
	end

	@testset "Lanczos scaling as Pillow scales" begin
		got = [MedievalATR.resize(UInt8.(permutedims(reduce(hcat, c["pixels"]))'), c["width"], c["height"]) for c in FIXTURES["resize"]]
		@test got == [UInt8.(permutedims(reduce(hcat, c["scaled"]))') for c in FIXTURES["resize"]]
	end

	@testset "ALTO and PAGE lines as Kraken reads them" begin
		got = Dict(name => [(l.id, l.baseline, l.boundary, l.text) for l in read_page(joinpath(DATA, name)).lines]
			for name in keys(FIXTURES["xml"]))
		@test got == Dict(name => [(l["id"], points(l["baseline"]), points(l["boundary"]), l["text"]) for l in lines]
			for (name, lines) in FIXTURES["xml"])
	end

	@testset "lines cut, prepared and read as Kraken does" begin
		model = load_model(:catmus_medieval)
		page = read_image(joinpath(DATA, FIXTURES["page"]))
		for l in FIXTURES["lines"]
			cut = cut_line(page, points(l["baseline"]), points(l["boundary"]))
			x = prepare_line(cut, model.height)
			probs = probabilities(model, x)
			@test cut == read_image(joinpath(DATA, l["cut"]))
			@test (size(x, 2), vec(map(i -> i[1] - 1, argmax(probs; dims = 1))), transcription(decode(model, probs))) ==
				(l["tensor_width"], Vector{Int}(l["labels"]), l["text"])
			@test isapprox(sum(Float64, x), l["tensor_sum"]; rtol = 1e-6)
			@test isapprox(vec(maximum(probs; dims = 1)), Float64.(l["peaks"]); atol = 1e-4)
		end
		lines = [TextLine(l["id"], points(l["baseline"]), points(l["boundary"]), "") for l in FIXTURES["lines"]]
		@test [p.text for p in recognize(model, page, lines)] == [l["text"] for l in FIXTURES["lines"]]
	end
end
