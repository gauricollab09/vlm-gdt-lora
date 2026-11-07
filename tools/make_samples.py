"""Render a small sample of the dataset plus a contact sheet you can look at.

    python tools/make_samples.py --n 12 --out samples

Writes samples/images/*.png, samples/samples.jsonl and samples/contact_sheet.png
(each crop next to its prompt kind, rendered text and the JSON label the model
must produce). Uses gen_crops.py so the samples are exactly what training sees.
"""
import argparse, json, os, subprocess, sys, textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "DejaVuSans.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    subprocess.run([sys.executable, str(ROOT / "gen_crops.py"), "--n", str(a.n), "--out", str(out),
                    "--seed", str(a.seed), "--test_frac", "0"], check=True, cwd=ROOT)
    rows = [json.loads(l) for l in open(out / "train.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: (r["kind"], r["image"]))
    (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.remove(out / "train.jsonl"); os.remove(out / "test.jsonl")

    # contact sheet: one row per sample -> [crop | kind + text | JSON label]
    f_body, f_mono, f_head = (ImageFont.truetype(str(FONT), s) for s in (15, 13, 20))
    crop_w, text_w, json_w, pad = 300, 320, 520, 14
    W = crop_w + text_w + json_w + 4 * pad
    thumbs, heights = [], []
    for r in rows:
        im = Image.open(out / r["image"]).convert("RGB")
        scale = min(crop_w / im.width, 110 / im.height, 1.0)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        label = json.dumps(json.loads(r["answer"]), ensure_ascii=False, indent=1)
        lines = label.splitlines()
        heights.append(max(im.height, 16 * len(lines)) + 2 * pad)
        thumbs.append((im, lines))
    H = 70 + sum(heights)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, pad), f"vlm-gdt-lora dataset sample  ({len(rows)} crops, seed {a.seed})", fill="black", font=f_head)
    d.text((pad, pad + 30), "crop (model input)            kind / rendered text            JSON label (model target)", fill=(90, 90, 90), font=f_body)
    y = 70
    for r, (im, lines), h in zip(rows, thumbs, heights):
        d.line([(0, y), (W, y)], fill=(210, 210, 210))
        sheet.paste(im, (pad, y + pad))
        x2 = crop_w + 2 * pad
        d.text((x2, y + pad), r["kind"].upper(), fill=(200, 60, 20), font=f_body)
        shown = r["text"]
        if r["kind"] == "gdt":  # the symbol cell is drawn geometry; name it rather than print a glyph DejaVu may lack
            shown = json.loads(r["answer"])["characteristic"].replace("_", " ") + " |" + shown.split("|", 1)[1]
        d.text((x2, y + pad + 22), textwrap.shorten(shown, 40), fill="black", font=f_body)
        d.text((x2, y + pad + 46), Path(r["image"]).name, fill=(120, 120, 120), font=f_mono)
        x3 = crop_w + text_w + 3 * pad
        for i, ln in enumerate(lines):
            d.text((x3, y + pad + 16 * i), ln, fill="black", font=f_mono)
        y += h
    sheet.save(out / "contact_sheet.png")
    print("wrote", out / "contact_sheet.png", f"{W}x{H}")

    # legend: all 14 GD&T characteristic symbols as the generator draws them
    sys.path.insert(0, str(ROOT))
    import gen_crops
    s, cell_w, cell_h, cols = 48, 300, 84, 2
    names = list(gen_crops.GDT)
    rows_n = (len(names) + cols - 1) // cols
    leg = Image.new("RGB", (cols * cell_w + 2 * pad, 50 + rows_n * cell_h), "white")
    d = ImageDraw.Draw(leg)
    d.text((pad, pad), "GD&T characteristic symbols (ASME Y14.5), as rendered", fill="black", font=f_head)
    for i, name in enumerate(names):
        sym, needs_datums = gen_crops.GDT[name]
        cx = pad + (i % cols) * cell_w
        cy = 50 + (i // cols) * cell_h
        d.rectangle([cx, cy, cx + s + 16, cy + s + 16], outline="black", width=2)
        gen_crops.draw_symbol(d, sym, cx + 8, cy + 8, s, lw=3)
        d.text((cx + s + 30, cy + 6), name.replace("_", " "), fill="black", font=f_body)
        d.text((cx + s + 30, cy + 28), f"label symbol {sym}  U+{ord(sym):04X}", fill=(90, 90, 90), font=f_mono)
        d.text((cx + s + 30, cy + 46), "datums required" if needs_datums else "no datums (form)", fill=(120, 120, 120), font=f_mono)
    leg.save(out / "gdt_symbols_legend.png")
    print("wrote", out / "gdt_symbols_legend.png")


if __name__ == "__main__":
    main()
