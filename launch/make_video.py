"""Render the purgekv launch video (1920x1080, 30 fps, H.264) from the measured results.

  python launch/make_video.py [out.mp4] [preview_dir]
Needs: pillow, imageio-ffmpeg. Every number below comes from results/ (see README).
"""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H, FPS = 1920, 1080, 30
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "purgekv_launch.mp4")
PREVIEW = sys.argv[2] if len(sys.argv) > 2 else None

BG, FG, MUTED, LINE = (14, 17, 22), (242, 244, 247), (138, 148, 166), (40, 46, 56)
RED, GREEN, BLUE, AMBER = (229, 83, 75), (63, 185, 132), (91, 141, 239), (224, 168, 79)

_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
        try:
            # axes: Width, Optical Size, GRAD, Weight
            f.set_variation_by_axes([100, min(96, max(17, size * 0.75)), 400, 700 if bold else 400])
        except Exception:
            pass
        _fonts[k] = f
    return _fonts[k]


def ease(x):  # ease-out cubic, clamped
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def mix(c, a):  # fade colour toward background
    return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


class Scene:
    def __init__(self, dur, fn):
        self.dur, self.fn = dur, fn


def T(d, xy, s, size, color=FG, bold=False, a=1.0, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=mix(color, a), anchor=anchor)


def tag(d, x, y, s, color, a):
    f = font(26, True)
    w = d.textlength(s, font=f)
    d.rounded_rectangle([x, y, x + w + 36, y + 48], 24, outline=mix(color, a), width=3)
    d.text((x + 18, y + 24), s, font=f, fill=mix(color, a), anchor="lm")


def hbars(d, t, rows, x0, y0, maxv, unit, bar_w=1000, gap=118, start=0.6):
    """rows: (label, value, color, value_text or None). Animated grow + count-up."""
    for i, (label, v, col, vt) in enumerate(rows):
        p = ease((t - start - i * 0.35) / 1.1)
        a = ease((t - start - i * 0.35) / 0.5)
        y = y0 + i * gap
        T(d, (x0, y), label, 34, FG, a=a)
        d.rounded_rectangle([x0, y + 50, x0 + bar_w, y + 92], 10, fill=mix(LINE, a))
        if p > 0:
            d.rounded_rectangle([x0, y + 50, x0 + max(14, bar_w * v / maxv * p), y + 92], 10, fill=mix(col, a))
        shown = vt if (vt and p >= 0.999) else f"{v * p:.1f} {unit}" if v * p < 100 else f"{v * p:,.0f} {unit}"
        T(d, (x0 + bar_w + 30, y + 71), shown, 38, col, True, a, "lm")


def header(d, t, kicker, title, kcolor=MUTED):
    a = ease(t / 0.5)
    T(d, (140, 120), kicker, 30, kcolor, True, a)
    T(d, (140, 170), title, 60, FG, True, a)


# ---------- scenes ----------

def s_title(d, t):
    a = ease(t / 0.8)
    T(d, (W / 2, 400), "Your local LLM is quietly", 84, FG, True, a, "mm")
    T(d, (W / 2, 505), "swapping your Mac to SSD.", 84, FG, True, a, "mm")
    b = ease((t - 1.2) / 0.8)
    T(d, (W / 2, 640), "We measured it. Here's what actually fixes it.", 40, MUTED, a=b, anchor="mm")
    T(d, (W / 2, 960), "M4 MacBook · 16 GB · MLX 0.32 · real benchmarks", 28, MUTED, a=b, anchor="mm")


def s_setup(d, t):
    header(d, t, "THE TEST", "12 idle chats, then you open a big app")
    for i in range(12):
        a = ease((t - 0.5 - i * 0.08) / 0.4)
        x, y = 160 + (i % 6) * 120, 380 + (i // 6) * 120
        d.rounded_rectangle([x, y, x + 96, y + 96], 16, fill=mix(BLUE, a))
    a = ease((t - 0.7) / 0.5)
    T(d, (160, 650), "12 chats × 8K tokens", 40, FG, True, a)
    T(d, (160, 705), "= 3 GB of KV cache sitting idle", 34, MUTED, a=a)
    b = ease((t - 2.0) / 0.6)
    grow = ease((t - 2.0) / 1.2)
    d.rounded_rectangle([1100, 380, 1100 + 600 * grow + 1, 600], 20, fill=mix(AMBER, b))
    T(d, (1100, 650), "Then open an 8 GB app", 40, FG, True, b)
    T(d, (1100, 705), "What does macOS do with that 3 GB?", 34, MUTED, a=b)


def s_problem(d, t):
    header(d, t, "THE PROBLEM", "Data swapped to SSD as the app opens", RED)
    hbars(d, t, [("Idle caches kept in RAM (how mlx-lm works today)", 4.4, RED, "4.4 GB"),
                 ("Idle caches moved to an SSD file", 0.07, GREEN, "0.07 GB")],
          140, 330, 4.4, "GB", gap=150)
    a = ease((t - 2.6) / 0.6)
    T(d, (140, 720), "Average of 3 runs. In one run the app then needed 11.6 s", 34, MUTED, a=a)
    T(d, (140, 770), "to touch its own memory, instead of ~1 s.", 34, MUTED, a=a)


def s_idea(d, t):
    header(d, t, "THE CLEVER IDEA", "Let the macOS kernel delete idle caches for free")
    a = ease((t - 0.4) / 0.5)
    T(d, (140, 290), "Purgeable memory: the kernel drops it under pressure, no swap.", 36, MUTED, a=a)
    T(d, (140, 340), "We found no Mac LLM runtime using it for KV caches. So we built it.", 36, MUTED, a=a)
    for i in range(12):
        x, y = 160 + (i % 6) * 120, 440 + (i // 6) * 120
        lost = ease((t - 2.2 - i * 0.12) / 0.25)
        col = tuple(int(BLUE[k] + (RED[k] - BLUE[k]) * lost) for k in range(3))
        d.rounded_rectangle([x, y, x + 96, y + 96], 16, fill=mix(col, a))
    b = ease((t - 4.0) / 0.6)
    T(d, (1000, 470), "The kernel purged them before", 44, FG, True, b)
    T(d, (1000, 530), "macOS even raised a warning.", 44, FG, True, b)
    T(d, (1000, 610), "12 / 12 chats lost", 40, RED, True, b)
    T(d, (1000, 660), "130–143 s to recompute them", 36, MUTED, a=b)
    tag(d, 160, 760, "NEGATIVE RESULT · 3 of 3 runs", RED, b)


def s_works(d, t):
    header(d, t, "WHAT WORKS", "SSD tier: new tokens only + 8-bit", GREEN)
    T(d, (140, 260), "Total written to SSD · 4 chats, 9 saves each · 3B model", 32, MUTED, a=ease(t / 0.5))
    hbars(d, t, [("Rewrite the whole cache every save", 20.2, RED, "20.2 GB"),
                 ("Write only new tokens", 2.7, AMBER, "2.7 GB"),
                 ("+ lossless compression", 2.3, BLUE, "2.3 GB"),
                 ("+ 8-bit storage (lossy: can change outputs)", 1.4, GREEN, "1.4 GB")],
          140, 330, 20.2, "GB", gap=135)
    tag(d, 1480, 895, "14× less", GREEN, ease((t - 2.4) / 0.5))


def s_resume(d, t):
    header(d, t, "AND IT'S FAST", "Resume a 6K-token chat, cold from SSD")
    hbars(d, t, [("Recompute from scratch", 19.2, RED, "19.2 s"),
                 ("SSD tier, lossless (new tokens only)", 0.50, GREEN, "0.50 s")],
          140, 330, 19.2, "s", gap=150)
    a = ease((t - 2.2) / 0.6)
    T(d, (140, 720), "38× faster · bit-identical output", 44, FG, True, a)
    T(d, (140, 780), "Streaming the cache layer by layer cuts short-turn resumes a further 1.6×.", 32, MUTED, a=a)


def s_studio(d, t):
    header(d, t, "SCALED UP (ESTIMATE)", "70B coding agent on a Mac Studio", AMBER)
    T(d, (140, 260), "SSD writes per year · 20 sessions/day growing to 32K tokens", 32, MUTED, a=ease(t / 0.5))
    hbars(d, t, [("Naive SSD cache (rewrite every save)", 1293, RED, "1,293 TB"),
                 ("New tokens only + 8-bit", 42, GREEN, "42 TB")],
          140, 330, 1293, "TB", gap=150)
    a = ease((t - 2.4) / 0.6)
    T(d, (140, 700), "Typical SSDs are rated for ~600–1,200 TB written.", 36, FG, a=a)
    T(d, (140, 750), "Mac SSDs are soldered to the board.", 36, FG, a=a)
    T(d, (140, 850), "Extrapolated from measured ratios + published model sizes; not run on a Studio.", 28, MUTED, a=a)


def s_cta(d, t):
    a = ease(t / 0.7)
    T(d, (W / 2, 330), "purgekv", 110, FG, True, a, "mm")
    T(d, (W / 2, 440), "Open benchmarks for LLM memory on the Mac", 44, MUTED, a=a, anchor="mm")
    items = ["Raw data for every run", "Negative results included", "Reproduce it: ./run_all.sh", "MIT licensed"]
    for i, s in enumerate(items):
        b = ease((t - 1.0 - i * 0.3) / 0.5)
        y = 560 + i * 70
        d.ellipse([720, y - 9, 738, y + 9], fill=mix(GREEN, b))
        T(d, (760, y), s, 38, FG, a=b, anchor="lm")


SCENES = [Scene(4.5, s_title), Scene(5.5, s_setup), Scene(7.0, s_problem), Scene(8.0, s_idea),
          Scene(7.5, s_works), Scene(6.0, s_resume), Scene(8.0, s_studio), Scene(6.0, s_cta)]
FADE = 0.4


def render():
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", OUT],
                         stdin=subprocess.PIPE)
    for si, sc in enumerate(SCENES):
        n = int(sc.dur * FPS)
        for f in range(n):
            t = f / FPS
            img = Image.new("RGB", (W, H), BG)
            sc.fn(ImageDraw.Draw(img), t)
            # scene fade in/out
            k = min(1.0, t / FADE, (sc.dur - t) / FADE)
            if k < 1:
                img = Image.blend(Image.new("RGB", (W, H), BG), img, max(0.0, k))
            # progress bar
            ImageDraw.Draw(img).rectangle([0, H - 6, W * (si + t / sc.dur) / len(SCENES), H], fill=LINE)
            p.stdin.write(img.tobytes())
            if PREVIEW and f == int(n * 0.85):
                img.save(os.path.join(PREVIEW, f"scene{si + 1}.png"))
    p.stdin.close()
    p.wait()
    print(OUT, f"{sum(s.dur for s in SCENES):.1f}s")


if __name__ == "__main__":
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
    render()
