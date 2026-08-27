# Llama 3.2 Vision C++ / Hugging Face parity

This folder preserves the findings from the August 2026 parity investigation.
The detailed, reverse-chronological investigation is in [`debug-log.md`](debug-log.md).

## Headline result

Hugging Face "one tile" and the C++ one-tile path originally processed
different physical vision sequences. Transformers used padded 1,608-token tile
slots, including inactive slots, while C++ used one compact 1,601-token tile.
Making Hugging Face physically compact substantially improved parity. Following
Meta's prompt order by placing `<|image|>` before the user text further improved
50-example agreement between compact Hugging Face and C++ from `0.70` to `0.84`.

## 50-example evaluation

This BF16 VQA evaluation uses the same 50 examples for every backend. Agreement
compares normalized generated answers. Both C++ variants use the unformatted
user-text API and build their own single-BOS chat prompt; they differ only in
image-token placement.

- **HF one active tile:** the input contains one real image tile, but the
  internal vision state retains four tile slots. Each slot is padded to 1,608
  tokens, giving a 6,432-token internal sequence containing one active tile and
  three inactive slots.
- **HF compact one tile:** the input still contains one real image tile, but the
  vision forward is modified to keep only that tile internally. It uses the
  same compact 1,601-token sequence as C++, with no alignment padding or
  inactive tile slots.
- **C++:** both C++ variants always use one compact 1,601-token tile internally.
  They differ only in whether `<|image|>` appears before or after the user text.

### Accuracy

| Backend | Accuracy |
|---|---:|
| HF one active tile (four internal slots) | 0.8000 |
| HF compact one tile (one internal slot) | 0.6400 |
| C++ image before user text | 0.6867 |
| C++ original image placement | 0.6400 |

### Agreement

| | HF one active tile | HF compact one tile | C++ image first | C++ original |
|---|---:|---:|---:|---:|
| HF one active tile | 1.00 | 0.70 | 0.74 | 0.66 |
| HF compact one tile | 0.70 | 1.00 | 0.84 | 0.70 |
| C++ image first | 0.74 | 0.84 | 1.00 | 0.70 |
| C++ original | 0.66 | 0.70 | 0.70 | 1.00 |

### Findings

- Moving the C++ image token before the user text improves agreement with
  compact HF from `0.70` to `0.84` and with padded HF from `0.66` to `0.74`.
- Image-first C++ is the closest implementation to compact HF, but the remaining
  `0.16` disagreement shows that prompt and compact-sequence semantics do not
  eliminate all numerical output differences over 50 examples.
- Padded HF remains the most accurate on this VQA sample (`0.80`). Compact HF
  and original-placement C++ both score `0.64`; image-first C++ scores `0.6867`.
- Exact generated token IDs match on 38/50 examples between compact HF and
  image-first C++, versus 24/50 between compact HF and original-placement C++.

## Provenance

- Model: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- Hugging Face revision: `9eb2daaa8597bf192a8b0e73f848f3a102794df5`
- Source investigation branch: `cpp-parity`, based on commit `c843153`
- Isolated production fix: commit `5343090` on
  `fix/vlm-image-prompt-order`, based on `main` commit `7584cb8`
- Raw JSONL outputs, HTML explorers, images, and `/tmp` activation dumps were
  intentionally not copied into this repository.

The numerical results are diagnostic rather than a general model-quality
benchmark: they use the first 50 examples from the local VQA evaluation path.
