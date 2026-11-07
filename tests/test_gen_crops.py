import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def test_no_duplicate_crops_and_no_train_test_overlap(tmp_path):
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(ROOT / "gen_crops.py"), "--n", "300", "--out", str(out), "--seed", "0"],
        check=True, cwd=ROOT,
    )
    train, test = _rows(out / "train.jsonl"), _rows(out / "test.jsonl")
    assert len(train) + len(test) == 300

    # every row carries the rendered text so a reader can see what the image says
    assert all("text" in r for r in train + test)

    texts = [r["text"] for r in train + test]
    assert len(texts) == len(set(texts)), "duplicate crop text within the dataset"

    overlap = {r["text"] for r in train} & {r["text"] for r in test}
    assert not overlap, f"train/test share rendered crops: {sorted(overlap)[:5]}"

    # every image file referenced actually exists
    for r in train + test:
        assert (out / r["image"]).exists()


def test_all_14_gdt_symbols_render_distinct_and_non_blank():
    sys.path.insert(0, str(ROOT))
    import gen_crops
    from PIL import Image, ImageDraw

    renders = {}
    for name, (sym, _) in gen_crops.GDT.items():
        img = Image.new("RGB", (40, 40), "white")
        gen_crops.draw_symbol(ImageDraw.Draw(img), sym, 0, 0, 40)
        ink = sum(1 for p in img.convert("L").tobytes() if p < 128)
        assert ink > 20, f"{name} drew (almost) nothing"
        renders[name] = img.tobytes()
    assert len(set(renders.values())) == len(renders), "two symbols render identically"
