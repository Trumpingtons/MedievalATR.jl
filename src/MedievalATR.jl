"""
	MedievalATR

Automated text recognition of medieval manuscripts in Julia: Kraken's recognition models, such as CATMuS Medieval,
read from the files Kraken saves and computed as Kraken computes them.
"""
module MedievalATR

import ColorTypes
import Downloads
import EzXML
import FileIO
import FixedPointNumbers
import ImageIO
import JSON
import LinearAlgebra
import NNlib
import Scratch
import SHA

export Recognizer, Character, load_model, probabilities, decode, transcription, cut_line, prepare_line
export TextLine, Page, Prediction, read_page, read_image, recognize, write_predictions, model_path

include("protobuf.jl")
include("network.jl")
include("model.jl")
include("download.jl")
include("pillow.jl")
include("lines.jl")
include("xml.jl")
include("recognize.jl")

end
