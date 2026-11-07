"""
Synthetic engineering-drawing annotation crops with exact JSON labels.
Generates (1) dimension callouts and (2) GD&T feature control frames.
Output: out/images/*.png and out/train.jsonl, out/test.jsonl

Usage: python gen_crops.py --n 600 --out out
"""
import argparse, json, os, random, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

GDT = {  # characteristic -> unicode symbol, needs_datums
    "flatness": ("\u23e5", False),
    "straightness": ("\u23e4", False),
    "circularity": ("\u25cb", False),
    "cylindricity": ("\u232d", False),
    "perpendicularity": ("\u27c2", True),
    "parallelism": ("\u2225", True),
    "angularity": ("\u2220", True),
    "position": ("\u2316", True),
    "concentricity": ("\u25ce", True),
    "symmetry": ("\u232f", True),
    "profile_of_a_surface": ("\u2313", True),
    "profile_of_a_line": ("\u2312", True),
    "circular_runout": ("\u2197", True),
    "total_runout": ("\u2330", True),
}
DIA = "\u2300"


_FONT_WARNED = False


def load_font(size):
    # DejaVu has the GD&T glyphs; fall back to default if missing
    global _FONT_WARNED
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, "assets", "DejaVuSans.ttf"),           # vendored copy, works everywhere
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Debian/Ubuntu/Colab
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/opt/homebrew/share/fonts/DejaVuSans.ttf",               # macOS (brew install font-dejavu)
              os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf"),
              "/Library/Fonts/DejaVuSans.ttf",
              "DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    if not _FONT_WARNED:
        print("WARNING: DejaVuSans.ttf not found; GD&T symbols will render as boxes. "
              "Generate on Colab/Linux or install DejaVu.", file=sys.stderr)
        _FONT_WARNED = True
    return ImageFont.load_default()


def degrade(img):
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    if random.random() < 0.3:
        img = img.rotate(random.choice([90, 180, 270]), expand=True, fillcolor="white")
    if random.random() < 0.3:
        img = img.convert("L").point(lambda p: 255 if p > 160 else 0).convert("RGB")
    if random.random() < 0.15:
        img = Image.eval(img, lambda p: 255 - p)
    return img


def make_dimension():
    kind = random.choice(["linear", "diameter", "radius"])
    nominal = round(random.uniform(2, 250), random.choice([0, 1, 2]))
    tol_style = random.choice(["sym", "asym", "fit", "none"])
    qty = random.choice([1, 1, 1, 2, 4, 6])
    prefix = {"linear": "", "diameter": DIA, "radius": "R"}[kind]
    txt = f"{qty}x " if qty > 1 else ""
    txt += f"{prefix}{nominal:g}"
    label = {"type": kind, "quantity": qty, "nominal": nominal, "unit": "mm",
             "upper": None, "lower": None, "fit": None}
    if tol_style == "sym":
        t = random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        txt += f" \u00b1{t:g}"
        label["upper"], label["lower"] = t, -t
    elif tol_style == "asym":
        u = random.choice([0.0, 0.05, 0.1, 0.2]); l = -random.choice([0.05, 0.1, 0.2, 0.3])
        txt += f" +{u:g}/{l:g}"
        label["upper"], label["lower"] = u, l
    elif tol_style == "fit":
        fit = random.choice(["H7", "h6", "g6", "H8", "k6", "p6"])
        txt += f" {fit}"
        label["fit"] = fit
    return txt, label, "dimension"


def make_fcf():
    char = random.choice(list(GDT))
    sym, needs_datums = GDT[char]
    zone_dia = char in ("position", "cylindricity", "concentricity") and random.random() < 0.8
    tol = random.choice([0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2])
    mod = random.choice(["", "", "M", "L"]) if needs_datums else ""
    datums = []
    if needs_datums:
        for d in random.sample(list("ABCD"), random.randint(1, 3)):
            dm = random.choice(["", "", "(M)"])
            datums.append(d + dm)
    cells = [sym, f"{DIA if zone_dia else ''}{tol:g}{'(' + mod + ')' if mod else ''}"] + datums
    label = {"characteristic": char, "symbol": sym, "tolerance": tol,
             "zone": "diameter" if zone_dia else "width",
             "material_modifier": mod or None, "datums": datums}
    return cells, label, "gdt"


def render_text(txt, font):
    pad = 14
    w, h = font.getbbox(txt)[2:]
    img = Image.new("RGB", (w + 2 * pad, h + 2 * pad), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, pad // 2), txt, fill="black", font=font)
    # dimension line underneath, like a drawing
    y = h + pad + 4
    d.line([(4, y), (w + 2 * pad - 4, y)], fill="black", width=1)
    return img


def draw_symbol(d, sym, x, y, s, lw=2):
    """Draw a GD&T characteristic symbol as line geometry inside the s×s box at (x, y).

    No free font covers all 14 ASME Y14.5 symbols (DejaVu lacks 8, Noto lacks 3), and
    real drawings render them as CAD geometry anyway, so they are drawn here by hand.
    """
    def P(u, v):  # unit coords -> pixels
        return (x + u * s, y + v * s)
    L = lambda *pts: d.line([P(*p) for p in pts], fill="black", width=lw)
    C = lambda u0, v0, u1, v1: d.ellipse([P(u0, v0), P(u1, v1)], outline="black", width=lw)
    def arrow(u0, v0, u1, v1):
        L((u0, v0), (u1, v1))
        d.polygon([P(u1, v1), P(u1 - 0.22, v1 + 0.04), P(u1 - 0.04, v1 + 0.22)], fill="black")
    if sym == GDT["flatness"][0]:            # parallelogram
        L((0.12, 0.72), (0.38, 0.28), (0.88, 0.28), (0.62, 0.72), (0.12, 0.72))
    elif sym == GDT["straightness"][0]:      # horizontal line
        L((0.1, 0.5), (0.9, 0.5))
    elif sym == GDT["circularity"][0]:
        C(0.2, 0.2, 0.8, 0.8)
    elif sym == GDT["cylindricity"][0]:      # circle between two tangent verticals
        C(0.25, 0.25, 0.75, 0.75); L((0.12, 0.1), (0.12, 0.9)); L((0.88, 0.1), (0.88, 0.9))
    elif sym == GDT["perpendicularity"][0]:  # inverted T
        L((0.5, 0.15), (0.5, 0.8)); L((0.12, 0.8), (0.88, 0.8))
    elif sym == GDT["parallelism"][0]:       # two slanted lines
        L((0.25, 0.85), (0.55, 0.15)); L((0.5, 0.85), (0.8, 0.15))
    elif sym == GDT["angularity"][0]:        # angle
        L((0.88, 0.8), (0.12, 0.8), (0.75, 0.2))
    elif sym == GDT["position"][0]:          # circle with crosshair
        C(0.25, 0.25, 0.75, 0.75); L((0.5, 0.08), (0.5, 0.92)); L((0.08, 0.5), (0.92, 0.5))
    elif sym == GDT["concentricity"][0]:     # two concentric circles
        C(0.15, 0.15, 0.85, 0.85); C(0.33, 0.33, 0.67, 0.67)
    elif sym == GDT["symmetry"][0]:          # three bars, middle longest
        L((0.25, 0.3), (0.75, 0.3)); L((0.1, 0.5), (0.9, 0.5)); L((0.25, 0.7), (0.75, 0.7))
    elif sym == GDT["profile_of_a_surface"][0]:  # closed half-disc
        d.pieslice([P(0.15, 0.2), P(0.85, 0.9)], 180, 360, outline="black", width=lw)
        L((0.15, 0.55), (0.85, 0.55))
    elif sym == GDT["profile_of_a_line"][0]:     # open arc
        d.arc([P(0.15, 0.25), P(0.85, 0.95)], 180, 360, fill="black", width=lw)
    elif sym == GDT["circular_runout"][0]:       # one arrow at 45°
        arrow(0.2, 0.8, 0.8, 0.2)
    elif sym == GDT["total_runout"][0]:          # two arrows on a baseline
        L((0.08, 0.85), (0.92, 0.85)); arrow(0.15, 0.85, 0.55, 0.2); arrow(0.5, 0.85, 0.9, 0.2)
    else:
        raise ValueError(f"unknown GD&T symbol {sym!r}")


def render_fcf(cells, font):
    pad = 8
    h_text = font.getbbox("Ag")[3]
    sym_box = h_text  # symbol cell is square-ish, sized to the text height
    widths = [sym_box + 2 * pad] + [font.getbbox(c)[2] + 2 * pad for c in cells[1:]]
    h = h_text + 2 * pad
    img = Image.new("RGB", (sum(widths) + 2, h + 2), "white")
    d = ImageDraw.Draw(img)
    x = 1
    for i, (c, w) in enumerate(zip(cells, widths)):
        d.rectangle([x, 1, x + w, h], outline="black", width=2)
        if i == 0:
            draw_symbol(d, c, x + pad, 1 + pad, sym_box, lw=max(2, h_text // 12))
        else:
            d.text((x + pad, pad // 2), c, fill="black", font=font)
        x += w
    return img


PROMPT_DIM = "Read this dimension callout and return JSON with keys: type, quantity, nominal, unit, upper, lower, fit."
PROMPT_FCF = "Read this GD&T feature control frame and return JSON with keys: characteristic, symbol, tolerance, zone, material_modifier, datums."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="out")
    ap.add_argument("--test_frac", type=float, default=0.17)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows = []
    seen = set()   # rendered text; a crop string must appear once, so train/test never share one
    attempts = 0
    while len(rows) < a.n and attempts < a.n * 20:
        attempts += 1
        font = load_font(random.choice([22, 26, 30, 34]))
        if random.random() < 0.5:
            txt, label, kind = make_dimension()
            if txt in seen:
                continue
            img = render_text(txt, font)
            prompt = PROMPT_DIM
        else:
            cells, label, kind = make_fcf()
            txt = " | ".join(cells)
            if txt in seen:
                continue
            img = render_fcf(cells, font)
            prompt = PROMPT_FCF
        seen.add(txt)
        img = degrade(img)
        i = len(rows)
        fn = f"images/{kind}_{i:05d}.png"
        img.save(os.path.join(a.out, fn))
        rows.append({"image": fn, "kind": kind, "text": txt, "prompt": prompt,
                     "answer": json.dumps(label, ensure_ascii=False)})
    if len(rows) < a.n:
        print(f"WARNING: only {len(rows)} unique crops possible, asked for {a.n}", file=sys.stderr)
    random.shuffle(rows)
    k = int(len(rows) * a.test_frac)
    for name, part in [("test.jsonl", rows[:k]), ("train.jsonl", rows[k:])]:
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) - k} train / {k} test to {a.out}/")


if __name__ == "__main__":
    main()
