# MedievalATR's tests, against values computed by Pillow 13 and Kraken 7.2 with CATMuS Medieval and blla (test/data,
# made by fixtures.py from the Kraken reference): polygon masks, Lanczos scaling, ALTO and PAGE reading, the lines of a
# small drawn page as Kraken cuts, prepares and reads them, blla's network on that page, and the lines and regions
# Kraken finds on drawn maps, which stand in for the network's, with their ALTO. The models are downloaded the first
# time, CATMuS Medieval from Zenodo and blla from Kraken's repository.
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

	@testset "blla's network on a drawn page as PyTorch computes it with one thread" begin
		model = load_segmenter(:blla)
		network = FIXTURES["network"]
		input, _ = MedievalATR.segmentation_input(model, read_image(joinpath(DATA, FIXTURES["page"])))
		output = MedievalATR.network_output(model, input)
		classes, rows, cols = network["size"]
		@test size(output) == (cols, rows, classes)
		got = [output[c + 1, r + 1, k] for k in 1:classes, r in network["rows"], c in network["cols"]]
		expected = [Float32(network["logits"][k][i][j]) for k in 1:classes, i in eachindex(network["rows"]), j in eachindex(network["cols"])]
		# the convolutions and LSTMs add up in another order than PyTorch's, by some thousandths at most
		@test maximum(abs.(got .- expected)) < 0.01
	end

	@testset "lines and regions on drawn maps, and their ALTO, as Kraken finds and writes them" begin
		model = load_segmenter(:blla)
		case = FIXTURES["segment"]
		stacked = read_image(joinpath(DATA, case["maps"]))[1, :, :]
		width, height = size(stacked, 1), size(stacked, 2) ÷ 4
		maps = cat((Float32.(stacked[:, (k - 1) * height + 1:k * height]) ./ 255f0 for k in 1:4)...; dims = 3)
		grey = read_image(joinpath(DATA, case["page"]))[1, :, :]
		s = MedievalATR.segment_maps(model, maps, grey, case["width"], case["height"]; path = case["page"])
		@test ([(l.baseline, l.boundary) for l in s.lines], s.kinds, [r .- 1 for r in s.line_regions], [(r.kind, r.boundary) for r in s.regions]) ==
			([(points(l["baseline"]), points(l["boundary"])) for l in case["lines"]], [l["kind"] for l in case["lines"]],
				[Vector{Int}(l["regions"]) for l in case["lines"]], [(r["kind"], points(r["boundary"])) for r in case["regions"]])
		# the layout from the tags on (Kraken's description names Kraken), one element to a line, as Kraken's template
		# closes a TextBlock on the line of its Shape
		layout(file) = filter(!isempty, strip.(split(replace(split(read(file, String), "</Description>")[2],
			"</Shape></TextBlock>" => "</Shape>\n</TextBlock>"), '\n')))
		mktempdir() do dir
			path = write_alto(joinpath(dir, "page.xml"), s)
			@test layout(path) == layout(joinpath(DATA, case["alto"]))
			# Kraken's reader, like read_page, drops a point that repeats the one before it
			@test [(l.id, l.baseline, l.boundary) for l in read_page(path).lines] ==
				[(l.id, l.baseline, MedievalATR.adjacent_unique(l.boundary)) for l in s.lines]
		end
	end
end
