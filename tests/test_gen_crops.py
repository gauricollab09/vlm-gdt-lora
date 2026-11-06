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
