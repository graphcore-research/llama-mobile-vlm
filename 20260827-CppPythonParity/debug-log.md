# Parity debug log

Keep this log in reverse chronological order. Add new notes at the top in
sections headed `## ...`.

## 2026-08-26 — Clean prompt comparison shows no BPE discrepancy

- Reconstructed historical C++ prompt construction from the genuinely
  unformatted example-1 user text, rather than passing it the already rendered
  HF chat prompt as in the earlier ablation.
- HF produced 89 tokens: two leading BOS tokens and the image token at index 6,
  immediately after the user header. C++ produced 88 tokens: one leading BOS
  token and the image token at index 82, immediately after the user text.
- Moving the C++ image token before the user text makes its complete 88-token
  sequence exactly equal to HF after dropping HF's extra leading BOS. Every
  ordinary text token and chat-header token matches. There is no observed C++
  versus HF BPE-tokenization difference on this prompt.
- The earlier `0.75` prompt ablation deliberately exercised the literal old API
  on an already formatted prompt and therefore measured double chat wrapping;
  it does not measure the intended unformatted-prompt usage. A clean model-level
  ablation would separately test image-token position and the extra BOS.

## 2026-08-26 — Compact-mode ablations identify the minimal parity changes

- Held compact 1,601-token Hugging Face vision fixed and independently restored
  three historical C++ behaviors over the same 20 BF16 VQA examples. Temporary
  runtime switches were removed after collecting the results.
- Restoring C++ text tokenization/chat wrapping reduced normalized agreement
  from `1.00` to `0.75`. C++ accuracy became `0.65` versus compact HF `0.60`;
  five substantive outputs differed and only 7/20 token sequences were exact.
- Restoring tanh-approximate GELU retained `1.00` normalized agreement and equal
  `0.60` accuracy. Token sequences were exact on 19/20 examples.
- Restoring the old sort/Gumbel greedy sampling path also retained `1.00`
  normalized agreement and equal `0.60` accuracy. Token sequences were exact on
  17/20 examples; differences were non-substantive punctuation/EOS behavior.
- On this sample, exact HF prompt token IDs plus compact vision sequencing are
  the minimal changes needed for perfect normalized agreement. Compact vision
  sequencing alone, with historical C++ prompt construction, is still “quite
  close” at `0.75`. Exact GELU and stable greedy argmax are not needed for
  normalized answer parity here, although they improve deterministic token-level
  semantics and remain sensible correctness changes.

## 2026-08-26 — Compact HF mode restores parity

- Added `--compact-one-tile`, which keeps the Hugging Face image processor at
  one tile and replaces the vision forward path with the C++ sequence semantics:
  exactly 1,601 tokens, no seven-token alignment padding, no inactive tile
  slots, and no vision attention mask. Tap indices and multimodal-projector
  feature ordering remain unchanged.
- Ran 20 BF16 VQA examples in
  `7-compact-one-tile-bf16-20`. Compact Hugging Face and C++ both scored
  `0.6000` accuracy with normalized output agreement `1.0000`.
- Generated token IDs were exactly equal on 18/20 examples. On examples 2 and
  10, Hugging Face emitted the same answer followed by a period before EOS,
  while C++ emitted EOS immediately; normalized answers still agree.
- For comparison, padded four-slot HF previously scored `0.80` versus C++
  `0.64` with only `0.70` agreement over 50 BF16 examples. Matching physical
  vision sequence semantics therefore removes the general parity gap on this
  20-example experiment.

## 2026-08-26 — One active tile is still a four-slot HF sequence

- The parity adapter does request `max_image_tiles = 1` from the processor, but
  then `_pad_tiles` expands `pixel_values`, `aspect_ratio_mask`, and the
  cross-attention mask to the model configuration's four tile slots. Thus the
  run has one active tile but local vision attention still receives 6,432
  positions (`4 * 1608`); C++ receives 1,601.
- A full layer-0 capture on example 1 quantified the two effects. Recomputing
  the Hugging Face attention mix with only the 1,601 active tokens differs from
  the actual mix by relative RMS `0.43381`. Including the active tile's seven
  padding positions improves this to `0.31577`. Including all four 1,608-token
  tile slots reproduces the actual mix to `0.00168`.
- Therefore the `+7` padding contributes materially, but the three inactive
  tile slots are the larger remaining effect. They have nonzero Q/K/V after
  class and positional embeddings (inactive-tile K RMS was `0.5843`) and valid
  queries can attend them under the Hugging Face mask construction.

## 2026-08-26 — Fine activations isolate layer-0 attention amplification

- Captured three BF16 examples at every local and global vision layer, before
  attention, after the attention residual, and after the FFN residual. The 6.7
  GB of raw paired NPY files are intentionally outside the worktree at
  `/tmp/llama-mobile-parity-fine-activations/{python,cpp}/000000..000002`.
- The first local layer enters attention at mean relative RMS `0.0270`, exits
  attention at `0.1601`, and exits its FFN at `0.1804`. Its attention residual
  contribution differs by `0.3385`; this is the first sharp, consistent jump
  across all three samples.
- Local layers 1–13 fluctuate around `0.11–0.20` state error. Drift then grows
  through the deeper local stack: `0.2915` after layer 15, `0.4899` after layer
  23, `0.8089` after layer 30, and `0.7755` after layer 31. Global vision enters
  at `0.7474` and finishes at `1.0122`.
- Drilling into local layer 0 gave mean relative RMS: normalized input `0.02469`,
  Q `0.00992`, K `0.01566`, V `0.02999`, attention mix `0.44166`, output
  projection `0.33842`, and post-residual state `0.16014`. The large discrepancy
  therefore first appears in the attention-weighted value mix.
- The C++ attention kernel itself is correct for the sequence it receives.
  Recomputing attention in PyTorch from the dumped C++ Q/K/V matches the C++
  mix to relative RMS `0.00165`, `0.00165`, and `0.00163` on examples 1–3.
- The concrete culprit is a sequence-format mismatch. C++ runs attention over
  only the active tile's 1,601 tokens. Hugging Face pads every tile to 1,608
  tokens and retains the model's four-tile layout (6,432 tokens) even when the
  processor is restricted to one active tile. Our comparison hooks trimmed
  those extra positions only while saving; they had already participated in
  Hugging Face attention.
- Hugging Face's `_prepare_aspect_ratio_attention_mask` constructs an outer
  product of the *invalid-position* indicator. Consequently a valid query to an
  inactive/padded key receives zero additive mask and is allowed; only pairs
  where both query and key are invalid receive the large negative mask. The
  inactive tile states are not generally zero after class/position embeddings.
  Thus the valid first tile attends thousands of states that do not exist in
  the C++ sequence, explaining the immediate 44% attention-mix gap.
- Later layers accumulate this first-layer format mismatch. The fix should make
  both implementations use the same tile/padding sequence and mask semantics;
  changing projector permutations or the attention arithmetic will not address
  it.

## 2026-08-26 — Projector inputs already diverge

- Dumped all five pre-layer projector taps and the final global-vision state for
  the same three BF16 examples. Hugging Face padding/tile layout was normalized
  to the valid `[1601, 1280]` first tile before comparison.
- Relative RMS at tap 3 was `0.107`, `0.108`, and `0.103` for examples 1–3.
  It grew to roughly `0.12` at tap 7, `0.23–0.26` at tap 15, `0.45–0.50` at tap
  23, and `0.54–0.58` at tap 30.
- The final global-vision projector input had relative RMS `1.033`, `1.040`, and
  `0.964`, with cosine similarity `0.489`, `0.478`, and `0.552` respectively.
- The multimodal projector is therefore consuming already-divergent states. Its
  permutation and serialized weights are correct; it is not the origin of the
  large discrepancy.
- Divergence accumulates through both the local vision transformer and especially
  the global vision transformer. The earliest large jump remains before/within
  local vision layer 0, so operator-level comparison there is the best next step.

## 2026-08-26 — Projector permutation is byte-exact

- Traced Hugging Face projector input layout: the first `1280` features are the
  final global-vision state; the remaining features are five tap states stacked
  on a new last dimension and flattened, so tap values are interleaved within
  each hidden dimension.
- The converter correctly unflattens the interleaved region as `[1280, 5]`,
  moves the tap dimension to the front, and appends the final-state block. This
  produces `[5 tap blocks, 1 final block]`, matching C++ accumulation order.
- Hugging Face records encoder hidden states before each layer. C++ also applies
  each configured tap before that layer, so tap indices `[3, 7, 15, 23, 30]`
  have the same semantics.
- Read the public BF16 SQT projector bytes directly and compared them with this
  transformation of the cached Hugging Face weight. Shapes were `[6, 4096,
  1280]`, and every byte matched.
- The projector-output discrepancy therefore comes from its input states and/or
  numerical accumulation, not an incorrect weight permutation. Next, compare
  every tap and each tap's projected contribution separately.

## 2026-08-26 — 5-coarse-activations localizes vision divergence

- Captured paired BF16 activations for the first three VQA examples. Examples 1
  and 3 disagree; example 2 is an agreeing control.
- Token embeddings are bit-exact for all three examples. Text layer 0 remains
  very close, with relative RMS error around `0.0031`.
- Vision patch embeddings already differ by about `0.016` relative RMS. After
  vision layer 0, relative RMS is `0.176`, `0.199`, and `0.166` for examples
  1–3. At vision layer 7 it is `0.124`, `0.135`, and `0.120`.
- The multimodal projection has the largest coarse discrepancy: relative RMS
  `0.918`, `0.928`, and `0.849`. Tile and padding layout were normalized to the
  same valid first-tile `[1601, ...]` representation before comparison.
- After the first cross-attention layer, text relative RMS is about `0.009–0.010`;
  by text layer 20 it is `0.036–0.040`.
- Final-logit relative RMS is `0.372` for disagreeing example 1, `0.133` for the
  agreeing control, and `0.256` for disagreeing example 3. The agreeing example
  therefore has materially closer final logits even though its early vision
  discrepancies are similar.
- The repeatable first large coarse divergence lies between the vision patch
  embedding and vision layer 0. The multimodal projection/tap accumulation is a
  second suspicious boundary and should be checked after isolating the first
  vision layer.

## 2026-08-26 — 4-image-input does not explain disagreement

- Dumped the BF16 C++ preprocessed image and Python `pixel_values` for VQA
  example 1, rearranging Python into C++ patch-major shape `[1600, 588]`.
- Against Python float32, C++ had mean absolute error `0.00635`, RMS error
  `0.01237`, relative RMS error `0.01044`, maximum error `0.19419`, and cosine
  similarity `0.999946`.
- After rounding Python to BF16, `709912/940800` elements matched exactly. The
  remaining difference had mean absolute error `0.00504`, RMS error `0.01221`,
  and maximum error `0.19531`; it is larger than BF16 rounding alone.
- Bypassed C++ image resizing/preprocessing by loading Python's patch-major NPY
  tensor through `loadNpy` and casting it to BF16.
- The output did not change: Python remained `5<|eot_id|>` and C++ remained
  `11<|eot_id|>`. Image preprocessing differences therefore do not explain the
  known example's output disagreement.
- The next likely boundary is the vision encoder or its multimodal projection,
  followed by cross-attention.

## 2026-08-26 — 1-text-only agrees

- Ran VQA example 1 through the BF16 text model with identical input token IDs,
  no image token, no image, and no cross-attention.
- Python and C++ both produced exact token IDs `[17, 128009]`, decoded as
  `2<|eot_id|>`.
- Normalized output agreement was `1.0`. Both VQA accuracy scores were zero, as
  expected when asking the image-dependent question without its image.
- This localizes the known example's disagreement to the vision preprocessing,
  vision encoder, or cross-attention path rather than the shared text-only path.

## 2026-08-26 — Debugging direction

- Current evidence points primarily to a general C++/PyTorch agreement issue,
  not an S3D8-only issue.
- Selected VQA example 1 (`How many computers?`) as the initial reproducer:
  Python BF16 outputs `5`, C++ BF16 outputs `11`, and C++ S3D8 outputs `10`.
- Planned investigation is documented in [debug-plan.md](debug-plan.md), starting
  with text-only BF16 and first-token logits before broad activation dumping.

## 2026-08-26 — 50-example results

- S3D8: Python accuracy `0.7200`, C++ accuracy `0.6267`, normalized output
  agreement `0.68`.
- BF16: Python accuracy `0.8000`, C++ accuracy `0.6400`, normalized output
  agreement `0.70`.
- BF16 improves agreement only slightly over S3D8. The remaining 30% BF16
  disagreement means compressed weights cannot be the sole cause.
- All four final JSONL streams were verified to contain 50 contiguous examples.
- A transient BF16 Python JSONL gap caused by a late preparation process was
  repaired by rerunning examples 21–50; final metrics were reproduced.

## 2026-08-26 — 20-example results

- S3D8: Python accuracy `0.75`, C++ accuracy `0.65`, normalized output agreement
  `0.65`.
- BF16: Python accuracy `0.80`, C++ accuracy `0.60`, normalized output agreement
  `0.75`.
- C++ BF16 and C++ S3D8 had exact token matches on `10/20` examples.
- Python BF16 and Python QAT had exact token matches on `15/20` examples.

## 2026-08-26 — Initial one-example run

- Python and C++ received identical input token IDs, including the duplicated
  leading BOS token.
- Python S3D8/QAT output `5`; C++ S3D8 output `10`.
- Python generation took `2.59s`; C++ generation took `20.27s`.
- The first end-to-end run took `1124.0s`, dominated by initial model and VQA
  dataset setup.

## 2026-08-26 — Environment and test notes

- Host is an NVIDIA GH200 on AArch64 with driver `570.148.08` and CUDA 12.8.
- Installed PyTorch `2.9.0+cu128`; an actual CUDA tensor operation succeeded.
- The parity Python tests pass.
- The C++ CLI builds successfully on the host after installing Clang and OpenMP.
- The full native test build currently fails on AArch64 because
  `#pragma float_control` is unsupported and promoted to an error by `-Werror`.
