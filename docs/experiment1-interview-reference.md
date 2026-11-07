# Experiment 1 Interview Reference

As of 2026-11-07. Editable copy: https://claude.ai/artifact/BNVMMEfWR4joQd8TD4Xcry

## The 30-second version

Say these four sentences in order; each later section backs one of them.

1. **Data.** I built a synthetic engineering-drawing data engine: a generator draws dimension callouts and GD&T feature control frames from a random label, so every crop has an exact JSON label. 600 crops, deduped, 498 train / 102 held out.
2. **Method.** QLoRA on Qwen2.5-VL-3B-Instruct on a free Colab T4: the vision encoder, projector and LLM frozen in 4-bit, rank-16 LoRA on every linear layer of the 36 decoder blocks (0.79% of parameters), one epoch of SFT with the loss on the answer tokens only.
3. **Evaluation.** A field-level benchmark: parse the model's JSON, flatten to (field, value) pairs, report JSON-valid rate, exact match, precision, recall, F1 and hallucination (1 − precision), split by crop type, on the same 102 crops before and after. Field F1 went 0.21 → 0.74, hallucination 0.78 → 0.26, in 5 minutes of training.
4. **Differently.** Error analysis showed GD&T exact match was zero because 8 of 14 symbols could not be rendered by any free font and all drew as the same box. I replaced font glyphs with drawn geometry, added a rendering test, and the rerun is next; then the vision-layer ablation, GRPO with the eval metric as reward, and real scanned pages.

## Numbers to have cold

| Area | Number | What it is |
| --- | --- | --- |
| Data | 600 | synthetic crops, 2 kinds (305 gdt, 295 dimension), exact labels |
| Data | 498 / 102 | train / held-out, no shared crop text |
| Data | 22 | vision tokens for a 300×60 crop (28 px per token) |
| Model | 3.78B | Qwen2.5-VL-3B-Instruct total parameters |
| Model | 4-bit NF4 | frozen base storage; fp16 compute; ~2 GB |
| LoRA | r = 16, α = 32 | rank and scale, α/r = 2; dropout 0.05 |
| LoRA | 7 × 36 | linears per layer (q k v o gate up down) × decoder layers |
| LoRA | 29.9M = 0.79% | trainable parameters, 75% of them in the MLPs |
| Training | 2e-4 | learning rate, ~10× full fine-tune; 10 warm-up steps, linear decay |
| Training | 2 × 4 = 8 | batch per device × grad accumulation = effective batch |
| Training | 63 steps, 5 min | 1 epoch on a T4; AdamW-8bit, weight decay 0.01 |
| Training | 4.56 → 0.04 | loss at step 5 → step 60; plateau ~0.05 from step 35 |
| Results | 0.21 → 0.74 | field F1, all 102 crops |
| Results | 0.78 → 0.26 | hallucination rate |
| Results | 0 → 29 | crops fully correct (53% of dimensions, 0% of gdt) |
| Results | 0.99 → 1.00 | JSON-valid rate; format was never the problem |

## Data: synthetic crops with exact labels

One random draw produces both the JSON label and the text that gets rendered, so the label is correct by construction and the data engine is free at any scale.

| Kind | Example crop | JSON keys |
| --- | --- | --- |
| dimension | `4x R97.78 ±0.2`, `⌀199 H7`, `79.5 H8` | type, quantity, nominal, unit, upper, lower, fit |
| gdt (feature control frame) | `◎ │ ⌀0.2(L) │ C(M) │ A(M)` | characteristic, symbol, tolerance, zone, material_modifier, datums |

- Degradation per crop at random: Gaussian blur (50%), rotation 90/180/270° (30%), binarise (30%), invert (15%). This keeps the task from being clean horizontal OCR.
- Dedupe on rendered text before the 83/17 split, so no crop string appears in both train and test. Nominal values are continuous, so the model cannot have memorised the test numbers.
- The crop is the model input; the JSON is the target. Prompt: "Read this GD&T feature control frame and return JSON with keys: …".
- Where to point: `gen_crops.py`, `samples/contact_sheet.png` (14 crops beside labels), `samples/gdt_symbols_legend.png`.

**Say:** "The label and the picture come from the same random draw, so the label is never wrong; I spent my effort on realism (degradation) and on hygiene (dedupe across the split)."

## Method: QLoRA SFT

Three decisions, each a sentence.

1. **QLoRA: freeze and quantize the base.** Qwen2.5-VL-3B-Instruct loaded with `load_in_4bit=True` (NF4, double quantization, fp16 compute). No gradient is stored for any base weight; the frozen weights are dequantized on the fly in forward and backward.
2. **LoRA on every linear layer of the LLM.** Adapters `W + (α/r)·B·A` on q, k, v, o, gate, up, down in all 36 decoder blocks; r = 16, α = 32, dropout 0.05. B initialised to zero, so step 0 is exactly the base model. Vision encoder and projector left frozen on purpose (the ablation flips `finetune_vision_layers=True`).
3. **SFT with the loss on the answer only.** Each row is a chat: user = [image, prompt], assistant = JSON. Cross-entropy on the assistant tokens; everything else labelled −100. LR 2e-4 (about 10× full fine-tuning), 10 warm-up steps, linear decay, AdamW-8bit, weight decay 0.01, effective batch 8, one epoch, fp16 autocast with loss scaling, gradient checkpointing.

Why those two rules (all linear layers, 10× LR): they are the findings of *LoRA Without Regret* (Thinking Machines, 2025) and match Unsloth's defaults. The parameter count checks out from the architecture: hidden 2048, MLP 11008, GQA k/v 256 → 16·(4096+2304+2304+4096+3·13056) = 831,488 per layer × 36 = 29,933,568.

**Say:** "QLoRA: the frozen base in 4-bit, rank-16 adapters on all seven projections of the 36 decoder layers, loss masked to the JSON, 2e-4 for one epoch. Five minutes on a T4."

## Evaluation: field-level benchmark, before vs after

Same 102 held-out crops, same prompt, greedy decoding, scored twice; the only thing that changed between the rows is the 0.79% LoRA block.

| Model | Kind | n | json_ok | exact | prec | rec | F1 | halluc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot | all | 102 | 0.99 | 0.00 | 0.22 | 0.20 | 0.21 | 0.78 |
| zero-shot | dimension | 57 | 1.00 | 0.00 | 0.34 | 0.33 | 0.33 | 0.66 |
| zero-shot | gdt | 45 | 0.98 | 0.00 | 0.05 | 0.04 | 0.05 | 0.95 |
| QLoRA | all | 102 | 1.00 | 0.29 | 0.74 | 0.74 | 0.74 | 0.26 |
| QLoRA | dimension | 57 | 1.00 | 0.53 | 0.85 | 0.85 | 0.85 | 0.15 |
| QLoRA | gdt | 45 | 1.00 | 0.00 | 0.60 | 0.58 | 0.59 | 0.40 |

How a score is computed (`eval_crops.py`): take the first `{…}` in the model text, `json.loads` it, flatten both prediction and label to a set of (path, value) pairs with numbers normalised, then precision = |P ∩ G| / |P|, recall = |P ∩ G| / |G|, exact = (P == G), hallucination = 1 − precision. Unparseable output counts every label field as missed.

- json_ok 0.99 → 1.00: the base already emits valid JSON; fine-tuning did not "teach JSON", it taught the schema and the reading.
- Hallucination 0.78 → 0.26: before, three of four fields the model wrote were not in the label. This is the number a manufacturing engineer cares about.
- Dimension exact 0.53: half the callouts perfectly read, including tolerances and ISO fits.
- gdt exact 0.00 with F1 0.59: four of six fields right on every frame, never the same two. See the next section.

**Say:** "I report per-field precision, recall, F1 and a hallucination rate, split by drawing type, on the same held-out crops before and after. Pooled F1 0.74 looked healthy; the per-kind split is what exposed the GD&T failure."

## The tofu bug: the error-analysis story

GD&T exact match stayed at zero because 8 of the 14 characteristic symbols were rendered as the same empty font box, so the characteristic was not in the pixels.

1. **Signal.** gdt F1 0.59 but exact 0.00 across 45 crops: the signature of "always wrong on the same two fields" (characteristic and symbol; 4 of 6 right ≈ 0.6).
2. **Look at the data.** A contact sheet of inputs beside labels showed frames like `▯ | 0.1(L) | D | C` for perpendicularity, position, total runout alike.
3. **Root cause.** The generator typed symbols as Unicode in DejaVu Sans. Checked with `fontTools`: DejaVu lacks 8 of 14; Noto Sans Math lacks 4; Noto Sans Symbols 2 lacks 12. No free font covers the set.
4. **Fix.** Draw the 14 symbols as line geometry (circle, parallelogram, inverted T, arrows) in `draw_symbol`, the way CAD renders them; text cells keep a vendored DejaVu for ⌀ and ±.
5. **Guard.** A test asserts all 14 render non-blank and pairwise distinct. Run 2 reuses the identical recipe on corrected data; `gdt exact` is the number to watch.

**Say:** "Per-kind metrics flagged it, a contact sheet showed it, fontTools confirmed it, geometry fixed it, a test keeps it fixed. It was a data bug, not a model limit, and the metric design is what caught it."

## What I would do differently, mapped to the JD

In the order I would run them; each is one flag or one small addition to the existing harness.

| Next | Change | Question it answers | JD line |
| --- | --- | --- | --- |
| Run 2 | drawn symbols, same recipe | does gdt exact leave zero | systematic error analysis |
| Run 3 | `finetune_vision_layers=True` | gain in the encoder or the LLM | vision encoder choices, PEFT vs full |
| Run 4 | `load_in_4bit=False` | QLoRA vs 16-bit LoRA on my own benchmark | parameter-efficient vs full fine-tuning |
| Run 5 | GRPO with `eval_crops.py` as reward | can a verifiable reward beat SFT on exact match | preference- or RL-based post-training |
| Then | DPO on engineer corrections | fix systematic over/under-calls cheaply | engineer corrections as training data |
| Then | per-field log-prob confidence + temperature scaling | route uncertain fields to review | calibrate model confidence |
| Then | real scanned pages, full-page context, crop-then-read vs whole page | resolution/token cost trade-off | high-resolution, text-dense images |
| Then | continued pretraining on drawing corpora | drawing conventions before task data | continued pretraining |

Also worth saying: the free-tier constraint was a feature. It forced QLoRA, a tiny image budget and a 5-minute loop, which made ablations cheap. With an uncapped cluster the recipe scales: full fine-tuning under FSDP or ZeRO-3, larger crops, and 100k+ synthetic pages from the same generator.

## Likely follow-up questions

- **Why LoRA rather than full fine-tuning?** Full FT of 3.8B needs fp16 weights, gradients and Adam states, about 60 GB; the T4 has 16. LoRA trains 0.79% and the adapter is the only artefact. With enough adapter capacity it matches full FT on this data size.
- **What does the Q add, and what does it cost?** The frozen base is stored in 4-bit NF4 (about 4× less memory); compute is still fp16 after dequantization, so each step is slower than 16-bit LoRA. The QLoRA paper reports no measurable quality loss; I have not measured it myself yet (run 4).
- **Why α = 32 with r = 16?** The update is scaled by α/r = 2; keeping α = 2r keeps the effective step stable if rank changes.
- **Why 2e-4?** B starts at zero so the adapter's effect grows from nothing; empirically LoRA's best LR is about 10× full fine-tuning's, nearly independent of rank.
- **Why mask the loss to the answer?** Prompt and image are given at inference; scoring them wastes gradient on text the model never produces and dilutes the JSON signal.
- **Why one epoch?** Loss plateaued by step 35 of 63; the data is templated, so a second pass mostly memorises. The held-out set is the check.
- **How do you know it is not memorising?** Test crops share no rendered text with training, nominal values are continuous, and 29 perfect crops on unseen numbers cannot come from lookup.
- **Why not train the vision encoder?** The zero-shot model saw the glyphs but did not know the schema; that is the cheaper hypothesis, and the ablation measures it.
- **How does backprop work through frozen 4-bit weights?** The gradient to a layer's input is Wᵀ times the output gradient, so frozen W is used in backward and dequantized again; no dL/dW is ever allocated, only dL/dA and dL/dB.
- **What is "hallucination" here?** 1 − precision over flattened (field, value) pairs: the share of fields the model wrote that are not in the label. 78% → 26%.
- **How many vision tokens is a crop?** A 300×60 crop resizes to 308×56, 88 patches of 14 px, merged 2×2 to 22 tokens. A full A1 page at readable resolution is thousands, which is the resolution-versus-cost question.
- **Was the run reproducible?** Yes: a re-run gave the identical zero-shot table and training loss equal to the fourth decimal (seeded data and LoRA init, greedy decoding).
- **What did the serving side look like?** Separate project, `t4-serving-bench`: vLLM on the same T4, measuring the token cost of images for Qwen2-VL-2B and Phi-3.5-vision.

## Links

- Repo: https://github.com/gauricollab09/vlm-gdt-lora (private; README has the results table and the contact sheet)
- Local: `~/AI_proj/vlm-gdt-lora/` with `samples/contact_sheet.png`, `samples/gdt_symbols_legend.png`, `notebooks/qwen_vl_lora.ipynb`
- Block diagram: https://claude.ai/artifact/T4NhqvNyNZrhYCXThUipVi
- Training strategy sheet: https://claude.ai/artifact/2edHbXXm31ayDo3h1LpgM8
- Mechanics, step by step: https://claude.ai/artifact/JPxC5t1oEWdXcExy5DFsFW
- Editable copy of this document: https://claude.ai/artifact/BNVMMEfWR4joQd8TD4Xcry
