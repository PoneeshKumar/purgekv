"""Render README / GitHub assets from the launch video's scenes.

  python launch/make_assets.py        # writes docs/assets/{streaming.gif,timeline.png,social-preview.png}
"""
import os
from PIL import Image
import make_video2 as V

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def scene(fn, t, dur, tint, bg_t=0.0):
    c = V.Canvas()
    fn(c, t, dur)
    return c.finish(V.background(bg_t + t, tint))


# 1. README hero GIF: the layer stack streaming, then the timeline comparison
frames = []
for i in range(int(5.4 * 12)):
    t = 0.9 + i / 12
    frames.append(scene(V.s_layers, t, 7.0, V.BLUE))
for i in range(int(4.4 * 12)):
    t = 0.2 + i / 12
    frames.append(scene(V.s_overlap, t, 5.6, V.GREEN))
frames += [frames[-1]] * 24  # hold the result for 2 s
small = [f.resize((800, 450), Image.LANCZOS).quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
         for f in frames]
small[0].save(os.path.join(OUT, "streaming.gif"), save_all=True, append_images=small[1:], duration=83, loop=0, optimize=True)

# 2. Static timeline diagram (end state of the comparison)
scene(V.s_overlap, 5.4, 5.6, V.GREEN).resize((1600, 900), Image.LANCZOS).save(os.path.join(OUT, "timeline.png"), optimize=True)

# 3. Social preview card (GitHub: Settings -> Social preview, 1280x640)
Y = 130  # shifts the whole card down so it sits centred after the crop
c = V.Canvas()
c.text((140, Y + 250), "purgekv", 150, V.FG, 1, 800, "ls", glow=True)
c.text((146, Y + 330), "Layer-streamed KV-cache resume for MLX on Apple Silicon", 46, V.MUTED, 1, 600, "ls")
c.text((146, Y + 520), "0.46 s", 92, V.MUTED, 1, 800, "ls")
c.d.line((146, Y + 488, 420, Y + 488), fill=V.col(V.RED, 0.85), width=7)
c.text((470, Y + 520), "→", 92, V.DIM, 1, 600, "ls")
c.text((600, Y + 520), "0.29 s", 92, V.GREEN, 1, 800, "ls", glow=True)
c.rect((1010, Y + 440, 1440, Y + 540), 50, V.GREEN, 1, glow=True)
c.text((1225, Y + 490), "1.6× faster", 54, V.BG0, 1, 800, "mm")
c.text((146, Y + 640), "Resume a 6K-token chat · M4 MacBook · bit-identical output", 34, V.DIM, 1, 500, "ls")
c.text((146, Y + 730), "github.com/PoneeshKumar/purgekv", 36, V.FG, 1, 600, "ls")
card = c.finish(V.background(3.0, V.GREEN)).crop((0, 60, 1920, 1020)).resize((1280, 640), Image.LANCZOS)
card.save(os.path.join(OUT, "social-preview.png"), optimize=True)

for f in sorted(os.listdir(OUT)):
    print(f, f"{os.path.getsize(os.path.join(OUT, f)) / 1e6:.2f} MB")
