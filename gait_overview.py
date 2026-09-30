"""All footfall diagrams of one species on one sheet: a row per walking brain, a column per command.

    python gait_overview.py        # -> docs/figures/gait/overview_<species>.png
"""
import os

from PIL import Image, ImageDraw, ImageFont

GAIT = "docs/figures/gait"
SHEETS = {
    "myrranox": (["myr_legs_v1", "myr_legs_v2", "myr_legs_v3", "myr_legs_v4", "myr_legs_v5"],
                 [("0.60", "walk 0.6 m/s"), ("1.20", "full speed 1.2 m/s"), ("0.00", "turn on the spot 72 deg/s")]),
    "dc": (["dc_legs_v1", "dc_legs_v2", "dc_legs_v3", "dc_legs_v4", "dc_legs_v5b", "dc_legs_v5c"],
           [("0.80", "walk 0.8 m/s"), ("2.50", "trot 2.5 m/s")]),
}
NOTES = {"myr_legs_v3": " (failed)", "myr_legs_v5": " (released)", "dc_legs_v5c": " (released)", "dc_legs_v5b": " (not used)"}
SCALE, LABEL_W, HEAD_H = 0.5, 230, 60


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    for species, (runs, cols) in SHEETS.items():
        sample = Image.open(os.path.join(GAIT, f"{runs[-1]}_footfall_{cols[0][0]}.png"))
        w, h = int(sample.width * SCALE), int(sample.height * SCALE)
        sheet = Image.new("RGB", (LABEL_W + w * len(cols), HEAD_H + h * len(runs)), "white")
        d = ImageDraw.Draw(sheet)
        for j, (_, title) in enumerate(cols):
            d.text((LABEL_W + j * w + 10, 18), title, fill="black", font=font(24))
        for i, run in enumerate(runs):
            d.text((10, HEAD_H + i * h + h // 2 - 14), run + NOTES.get(run, ""), fill="black", font=font(20))
            for j, (speed, _) in enumerate(cols):
                path = os.path.join(GAIT, f"{run}_footfall_{speed}.png")
                if os.path.exists(path):
                    sheet.paste(Image.open(path).convert("RGB").resize((w, h)), (LABEL_W + j * w, HEAD_H + i * h))
                else:
                    d.text((LABEL_W + j * w + w // 2 - 20, HEAD_H + i * h + h // 2), "-", fill="gray", font=font(28))
        out = os.path.join(GAIT, f"overview_{species}.png")
        sheet.save(out)
        print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
