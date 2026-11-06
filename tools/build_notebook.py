"""Build notebooks/qwen_vl_lora.ipynb from colab_qwen_vl_lora.py.

The .py is the source of truth (each '# %%' marker is a cell). Re-run this after
editing the script:  python tools/build_notebook.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "colab_qwen_vl_lora.py"
DST = ROOT / "notebooks" / "qwen_vl_lora.ipynb"

REPO = "https://github.com/<your-user>/vlm-gdt-lora.git"  # edit after pushing

SETUP = f'''# Colab T4: Runtime -> Change runtime type -> T4 GPU, then run top to bottom.
!nvidia-smi -L
!git clone -q {REPO}
%cd vlm-gdt-lora
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes
!python gen_crops.py --n 600 --out out
'''


def cells_from_script(text):
    parts = re.split(r"^# %% ?", text, flags=re.M)
    header, bodies = parts[0], parts[1:]
    cells = [md(header.strip().strip('"').strip())]
    for b in bodies:
        title, _, code = b.partition("\n")
        code = code.strip("\n")
        # lines like '# !pip ...' are shell commands that were commented out in the .py
        code = re.sub(r"^# (!|%)", r"\1", code, flags=re.M)
        if title.startswith(("1.", "2.")):
            continue  # replaced by SETUP cell
        cells.append(md(f"## {title.strip()}"))
        cells.append(py(code))
    return cells


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def py(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(True)}


def main():
    cells = cells_from_script(SRC.read_text(encoding="utf-8"))
    cells.insert(1, md("## 1–2. setup: GPU, clone, install, generate data"))
    cells.insert(2, py(SETUP))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                      "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                      "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    DST.parent.mkdir(exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", DST, len(cells), "cells")


if __name__ == "__main__":
    main()
