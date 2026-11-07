"""
Colab T4 experiment: Qwen2.5-VL-3B-Instruct, zero-shot vs QLoRA on synthetic drawing crops.
Run cell by cell in Colab (each '# %%' is a cell). GPU runtime required.

VERIFY BEFORE TRUSTING: the Unsloth API below follows Unsloth's public vision
fine-tuning notebooks (FastVisionModel, UnslothVisionDataCollator) as of 2025–26.
If a call errors, check https://unsloth.ai/docs for the current signature.
"""

# %% 1. install
# !pip install -q unsloth
# !pip install -q --no-deps trl peft accelerate bitsandbytes

# %% 2. data  (upload gen_crops.py + eval_crops.py to Colab first, or git clone your repo)
# !python gen_crops.py --n 600 --out out

# %% 3. load model (4-bit)
import json, torch
from unsloth import FastVisionModel
from PIL import Image
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()   # hides the per-call "max_new_tokens and max_length both set" notice

MODEL = "unsloth/Qwen2.5-VL-3B-Instruct"   # Unsloth mirror; fall back to "Qwen/Qwen2.5-VL-3B-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL, load_in_4bit=True, use_gradient_checkpointing="unsloth",
)

def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

train_rows = load_rows("out/train.jsonl")
test_rows = load_rows("out/test.jsonl")

# %% 4. inference helper + ZERO-SHOT baseline
def predict(row, max_new_tokens=128):
    img = Image.open("out/" + row["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(img, text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def run_eval(rows, out_path):
    FastVisionModel.for_inference(model)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"image": r["image"], "pred": predict(r)}, ensure_ascii=False) + "\n")
            if i % 20 == 0: print(i, r["image"])
    print("wrote", out_path)

run_eval(test_rows, "preds_zeroshot.jsonl")
# !python eval_crops.py --test out/test.jsonl --pred preds_zeroshot.jsonl

# %% 5. LoRA config  (language layers only first; set finetune_vision_layers=True for the ablation)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", random_state=0,
)

# %% 6. dataset in Unsloth's conversation format
def to_conv(r):
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": Image.open("out/" + r["image"]).convert("RGB")},
                                     {"type": "text", "text": r["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
    ]}
train_ds = [to_conv(r) for r in train_rows]

# %% 7. train
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_ds,
    args=SFTConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=2e-4, warmup_steps=10,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        output_dir="outputs", report_to="none",
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=1024,
    ),
)
stats = trainer.train()
print(stats)

# %% 8. FINE-TUNED eval
run_eval(test_rows, "preds_lora.jsonl")
# !python eval_crops.py --test out/test.jsonl --pred preds_lora.jsonl

# %% 9. save adapter
model.save_pretrained("lora_gdt_crops")
tokenizer.save_pretrained("lora_gdt_crops")
