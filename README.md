# MedievalATR.jl

Automated text recognition of medieval manuscripts in Julia. MedievalATR reads the models that
[Kraken](https://kraken.re) saves (`.mlmodel`), and does what Kraken 7.2 does with them:

- **Segmentation:** with blla, Kraken's default segmentation model, it finds the text lines and regions of a page
  image, the baseline and outline of each line, and their reading order, and writes them to an ALTO file.
- **Recognition:** with a recognition model such as CATMuS Medieval, it cuts out each line along its baseline and
  outline, prepares the line image for the network, runs the network and decodes the text.

It runs in plain Julia, with no Python and no PyTorch. It reads the page layout from ALTO and PAGE files, or finds it
itself.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Trumpingtons/MedievalATR.jl")
```

## Quick start

```julia
using MedievalATR

model = load_model(:catmus_medieval)        # downloaded from Zenodo the first time, then kept
page = read_page("folio_12r.xml")           # an ALTO or PAGE file with baselines and line polygons
for p in recognize(model, page)
    println(p.id, ": ", p.text)
end
```

`recognize` reads the lines on Julia's threads; start Julia with `-t auto` to use them all. A model file of your own is
read with `load_model("path/to/model.mlmodel")`.

A page without an ALTO or PAGE file is segmented first:

```julia
segmenter = load_segmenter(:blla)            # Kraken's default segmentation model, downloaded the first time
image = read_image("folio_12r.jpg")
s = segment(segmenter, image; path = "folio_12r.jpg")
write_alto("folio_12r.xml", s)               # the lines and regions, as Kraken writes them
for p in recognize(model, image, s.lines)
    println(p.id, ": ", p.text)
end
```

## Functions

| Function | What it does |
|---|---|
| `load_model(name)`, `load_model(path)` | A Kraken recognition model, by name (`:catmus_medieval`) or from its `.mlmodel` file |
| `load_segmenter(name)`, `load_segmenter(path)` | A Kraken segmentation model, by name (`:blla`) or from its `.mlmodel` file |
| `segment(segmenter, image)` | The lines and regions of a page image, each line with its baseline and outline, in reading order |
| `write_alto(path, segmentation)` | The lines and regions of a segmentation, written to an ALTO file as Kraken writes it |
| `read_page(path)` | The image and text lines of an ALTO or PAGE file, each with its baseline, polygon and transcription |
| `read_image(path)` | A page image as bytes (bands, width, height) |
| `recognize(model, page)`, `recognize(model, image, lines)` | The text of each line, as `Prediction`s with the line's id, the text and its characters |
| `cut_line(image, baseline, boundary)` | The image of one line, straightened along its baseline |
| `prepare_line(line, height)` | The network's input for a line image |
| `probabilities(model, x)` | The network's output, the probability of each label at each step |
| `decode(model, probs)` | The characters of the greedy CTC decoding, each with its steps and confidence |

## The same lines and text as Kraken

MedievalATR follows Kraken 7.2 step by step, and was checked against it on the 7,059 lines of the CMMHWR26 test set
(French, Latin, Spanish, Occitan and Czech):

- **Pages:** `read_page` gives every line the same id, baseline, polygon and transcription as Kraken's `XMLPage`.
- **Line images:** on the 706 lines Kraken's images were saved for, the cut lines and the network inputs are the same
  to the pixel. That holds for straight baselines (rotated) and for bent ones (warped piecewise).
- **Network:** the probabilities agree with PyTorch's to float32 precision.
- **Text:** `recognize` reads the same text as Kraken's `rpred` on all 7,059 lines, including the one line Kraken leaves
  empty.

`segment` follows Kraken 7.2's `blla.segment` step by step, and was checked against it on the 91 pages of the CMMHWR26
test set: from the page images, it finds the same 6,924 lines, with the same baselines, outlines, regions and reading
order, to the pixel. Its arithmetic is that of the libraries Kraken calls, as they are compiled for Apple silicon:
NumPy, SciPy, scikit-image, Pillow, Shapely with GEOS 3.13.1 (through LibGEOS.jl) and PyTorch, whose group
normalisation sums in an order that depends on the number of its threads (`torch_threads`, 1 by default, as
`kraken --threads 1`).

The line images are cut with Pillow's arithmetic as Pillow is compiled for Apple silicon, which fuses each product and
sum into one rounding. Kraken on a machine whose Pillow does not do this can differ from it by one grey level at a few
pixels on a polygon's edge.

The tests compare MedievalATR with values computed by Pillow 13 and Kraken 7.2: polygon masks, Lanczos scaling, ALTO
and PAGE reading, the lines of a small drawn page, cut, prepared and read, blla's network on that page, and the lines,
regions and ALTO file Kraken makes from drawn maps that stand in for the network's.

## The models

blla is Kraken's default segmentation model, published with Kraken under the Apache 2.0 licence. MedievalATR downloads
it from Kraken's repository, as Kraken 7.2 ships it, and checks its SHA-256 checksum.

CATMuS Medieval 1.6.2 (Pinche, Clérice and 12 coauthors, 2026, [10.5281/zenodo.21488839](https://doi.org/10.5281/zenodo.21488839))
is a Kraken model trained on Old and Middle French, Latin, Spanish (and other languages of Spain) and Italian. Its
transcriptions are graphematic: abbreviations are not expanded, and each diacritic is a character of its own (NFD). It
is published under the Creative Commons Attribution 4.0 licence. MedievalATR downloads it from Zenodo and checks its
SHA-256 checksum. It is not bundled with the package.

## Licence

MIT. The models MedievalATR downloads keep their own licences.
