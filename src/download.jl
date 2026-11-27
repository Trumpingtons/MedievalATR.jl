# The published models MedievalATR reads by name, downloaded once into a scratch space and checked against their
# SHA-256 checksums.

"""A published model: where it is downloaded from, its file name and its SHA-256 checksum."""
struct PublishedModel
	url::String
	file::String
	sha256::String
end

const MODELS = Dict(
	# CATMuS Medieval 1.6.2 (Zenodo 10.5281/zenodo.21488839, CC BY 4.0), whose file keeps the name of 1.6.0's weights
	:catmus_medieval => PublishedModel(
		"https://zenodo.org/api/records/21488839/files/catmus-medieval-1.6.0.mlmodel/content",
		"catmus-medieval-1.6.0.mlmodel",
		"227d10f15a936c2492105b7796a6152647c742f248af117856f417008eadbc9c"),
	# blla, Kraken's default segmentation model, as Kraken 7.2 ships it (Apache 2.0)
	:blla => PublishedModel(
		"https://raw.githubusercontent.com/mittagessen/kraken/7.2/kraken/blla.mlmodel",
		"blla.mlmodel",
		"a3ccc7309fe8df62d1cbd372706edc73db6a2a32806e3758522ed8528a734073"),
)

"""
	model_path(name::Symbol) -> String

The path of the published model `name` (`:catmus_medieval` or `:blla`), downloaded the first time it is asked for into
MedievalATR's scratch space and checked against its SHA-256 checksum.
"""
function model_path(name::Symbol)
	haskey(MODELS, name) || throw(ArgumentError("MedievalATR has no model named $(name); it has $(join(sort!(collect(keys(MODELS))), ", "))."))
	model = MODELS[name]
	path = joinpath(Scratch.@get_scratch!("models"), model.file)
	isfile(path) && return path
	partial = path * ".part"
	Downloads.download(model.url, partial)
	sum = bytes2hex(open(SHA.sha256, partial))
	if sum != model.sha256
		rm(partial; force = true)
		error("The download of $(model.file) has the SHA-256 checksum $(sum), not $(model.sha256).")
	end
	mv(partial, path; force = true)
	path
end

"""
	load_model(name::Symbol) -> Recognizer

The published model `name` (`:catmus_medieval`), downloaded the first time.
"""
load_model(name::Symbol) = load_model(model_path(name))

"""
	load_segmenter(name::Symbol) -> Segmenter

The published segmentation model `name` (`:blla`, Kraken's default), downloaded the first time.
"""
load_segmenter(name::Symbol) = load_segmenter(model_path(name))
