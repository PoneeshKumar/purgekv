"""Launch video v2: layer-streamed KV resume on Apple Silicon.

  python launch/make_video2.py OUT.mp4 VO_DIR [--preview DIR]
VO_DIR holds vo1.aiff ... vo8.aiff (macOS `say -v Samantha -r 172`, see launch/narration.txt).
Video: 1920x1080 @ 30 fps, Pillow frames -> H.264. Audio: narration + original synthesized
music and sound effects (numpy), ducked under the voice, AAC. Numbers are from results/stream_*.json.
"""
import math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1920, 1080, 30, 44100
OUT, VO_DIR = sys.argv[1], sys.argv[2]
PREVIEW = sys.argv[sys.argv.index("--preview") + 1] if "--preview" in sys.argv else None
ONLY_PREVIEW = "--frames-only" in sys.argv
NO_VOICE = "--no-voice" in sys.argv  # narration clips still set the scene timing

# ---------------- palette & type ----------------
BG0, BG1 = (9, 11, 16), (18, 22, 32)
FG, MUTED, DIM, LINE = (242, 244, 247), (150, 160, 178), (88, 96, 112), (44, 50, 62)
BLUE, GREEN, RED, AMBER, VIOLET = (94, 150, 255), (52, 211, 153), (244, 98, 92), (250, 182, 72), (167, 139, 250)
_fonts = {}


def font(size, weight=400):
    k = (size, weight)
    if k not in _fonts:
        f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", int(size))
        f.set_variation_by_axes([100, min(96, max(17, size * 0.75)), 400, weight])
        _fonts[k] = f
    return _fonts[k]


def clamp(x): return max(0.0, min(1.0, x))
def out3(x): x = clamp(x); return 1 - (1 - x) ** 3
def inout(x): x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def back(x):
    x = clamp(x); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def lerp(a, b, t): return a + (b - a) * t
def col(c, a=1.0): return (*c, int(255 * clamp(a)))


# ---------------- canvas with glow ----------------
class Canvas:
    def __init__(self):
        self.img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.g = ImageDraw.Draw(self.glow)

    def text(self, xy, s, size, c=FG, a=1.0, w=400, anchor="la", glow=False):
        self.d.text(xy, s, font=font(size, w), fill=col(c, a), anchor=anchor)
        if glow:
            self.g.text(xy, s, font=font(size, w), fill=col(c, a * 0.45), anchor=anchor)

    def rect(self, box, r, c, a=1.0, glow=False, outline=None, width=0):
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        self.d.rounded_rectangle(box, r, fill=None if outline else col(c, a), outline=col(outline, a) if outline else None, width=width)
        if glow:
            self.g.rounded_rectangle(box, r, fill=col(c, a * 0.8))

    def words(self, x, y, s, size, t, c=FG, w=700, stagger=0.06, dur=0.45, anchor_center=False, glow=False):
        """Kinetic text: each word rises and fades in."""
        f = font(size, w)
        parts = s.split(" ")
        widths = [self.d.textlength(p + " ", font=f) for p in parts]
        total = sum(widths) - self.d.textlength(" ", font=f)
        cx = x - total / 2 if anchor_center else x
        for i, (p, wd) in enumerate(zip(parts, widths)):
            k = out3((t - i * stagger) / dur)
            if k > 0:
                self.text((cx, y + (1 - k) * 28), p, size, c, k, w, "ls", glow)
            cx += wd

    def finish(self, bg):
        g = self.glow.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)
        g = Image.eval(Image.composite(g.convert("RGB"), Image.new("RGB", (W, H)), g.getchannel("A")), lambda v: int(v * 0.7))
        out = ImageChops.add(bg, g)
        out.paste(self.img, (0, 0), self.img)
        return out


# ---------------- background ----------------
_grad = None


def background(t, tint):
    global _grad
    if _grad is None:
        y = np.linspace(0, 1, H)[:, None, None]
        base = np.array(BG0) * (1 - y) + np.array(BG1) * y  # (H, 1, 3)
        _grad = np.broadcast_to(base, (H, W, 3)).astype(np.float32)
    arr = _grad.copy()
    # drifting soft spotlight in the scene's tint
    cx, cy = W * (0.5 + 0.18 * math.sin(t * 0.21)), H * (0.42 + 0.1 * math.cos(t * 0.17))
    yy, xx = np.ogrid[:H, :W]
    fall = np.exp(-(((xx - cx) / 900.0) ** 2 + ((yy - cy) / 620.0) ** 2))
    arr += fall[..., None] * np.array(tint, np.float32) * 0.13
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    off = (t * 14) % 60
    for gx in range(-60, W + 60, 60):
        for gy in range(-60, H + 60, 60):
            d.point((gx + off, gy + off * 0.5), fill=(40, 46, 60))
    return img


# ---------------- scenes ----------------
# each scene: (vo index, min duration, tint, draw fn(canvas, t, dur))

def chat_bubble(c, x, y, w, h, s, a, color=LINE, tc=FG, size=34):
    c.rect((x, y, x + w, y + h), 26, color, a)
    c.text((x + 30, y + h / 2), s, size, tc, a, 450, "lm")


def s_hook(c, t, dur):
    a = out3(t / 0.6)
    c.words(W / 2, 250, "Come back to a local AI chat…", 76, t, anchor_center=True)
    k = out3((t - 0.9) / 0.5)
    chat_bubble(c, 520, 380 + (1 - k) * 30, 880, 110, "Can you continue the refactor?", k, (36, 44, 60))
    k2 = out3((t - 1.8) / 0.5)
    if k2 > 0:
        x0, y0 = 520, 560
        c.rect((x0, y0, x0 + 880, y0 + 170), 26, (24, 29, 40), k2)
        spin = t * 6
        for i in range(8):
            ang = spin + i * math.pi / 4
            aa = k2 * (0.25 + 0.75 * ((i / 8 + t * 0.9) % 1))
            c.d.ellipse((x0 + 62 + 22 * math.cos(ang) - 5, y0 + 62 + 22 * math.sin(ang) - 5,
                         x0 + 62 + 22 * math.cos(ang) + 5, y0 + 62 + 22 * math.sin(ang) + 5), fill=col(AMBER, aa))
        c.text((x0 + 115, y0 + 62), "Reloading memory from SSD…", 34, AMBER, k2, 600, "lm")
        p = clamp((t - 2.0) / (dur - 2.6))
        c.rect((x0 + 40, y0 + 118, x0 + 840, y0 + 134), 8, LINE, k2)
        c.rect((x0 + 40, y0 + 118, x0 + 40 + 800 * inout(p), y0 + 134), 8, AMBER, k2, glow=True)


def lanes(c, x0, y0, wpx, a, label_y=None):
    for i, (name, cc) in enumerate((("SSD", BLUE), ("GPU", GREEN))):
        y = y0 + i * 120
        c.text((x0 - 30, y + 36), name, 34, cc, a, 700, "rm")
        c.rect((x0, y, x0 + wpx, y + 72), 14, (26, 31, 42), a)


def s_sequential(c, t, dur):
    c.words(160, 190, "Today: load everything, then compute.", 62, t)
    a = out3((t - 0.4) / 0.5)
    x0, y0, wpx = 380, 380, 1300
    lanes(c, x0, y0, wpx, a)
    scale = wpx / 0.5  # 0.5 s across
    load, comp = 0.30, 0.16  # measured: 0.46 s total, 0.16 s compute-only
    p = clamp((t - 0.8) / 3.4) * 0.46
    c.rect((x0, y0, x0 + scale * min(p, load), y0 + 72), 14, BLUE, a, glow=True)
    if p > load:
        c.rect((x0 + scale * load, y0 + 120, x0 + scale * p, y0 + 192), 14, GREEN, a, glow=True)
    if p < load and p > 0:  # GPU idle hatch
        for hx in range(int(x0), int(x0 + scale * p), 26):
            c.d.line((hx, y0 + 186, hx + 20, y0 + 126), fill=col(DIM, a * 0.8), width=3)
        c.text((x0 + scale * p / 2, y0 + 156), "idle", 30, MUTED, a, 600, "mm")
    ph = x0 + scale * p
    c.d.line((ph, y0 - 30, ph, y0 + 222), fill=col(FG, a * 0.8), width=3)
    c.text((ph, y0 - 44), f"{p:.2f} s", 30, FG, a, 600, "ms")
    k = out3((t - 4.3) / 0.5)
    c.text((x0, y0 + 300), "GPU waits for the SSD. Then the SSD waits for the GPU.", 38, MUTED, k)


def s_layers(c, t, dur):
    c.words(W / 2, 170, "So we stream it, layer by layer.", 66, t, anchor_center=True)
    a = out3((t - 0.3) / 0.5)
    n, x0, y0, lw, lh, gap = 28, 660, 250, 600, 18, 7
    cx = x0 + lw / 2
    prog = clamp((t - 0.9) / (dur - 1.8)) * (n + 1.5)
    readi, compi = prog, prog - 1.2
    for i in range(n):
        y = y0 + i * (lh + gap)
        c.rect((x0, y, x0 + lw, y + lh), 6, (28, 33, 45), a)
        if i < compi:
            c.rect((x0, y, x0 + lw, y + lh), 6, GREEN, a * 0.85, glow=abs(i - compi) < 2)
        elif i < readi:
            fill = clamp(readi - i)
            c.rect((x0, y, x0 + lw * fill, y + lh), 6, BLUE, a, glow=True)
    # SSD and GPU badges with arrows to the active rows
    ry = y0 + min(n - 1, max(0, int(readi))) * (lh + gap) + lh / 2
    gy = y0 + min(n - 1, max(0, int(compi))) * (lh + gap) + lh / 2
    c.rect((250, ry - 46, 470, ry + 46), 20, BLUE, a, glow=True)
    c.text((360, ry), "SSD read", 32, BG0, a, 700, "mm")
    c.d.line((470, ry, x0 - 14, ry), fill=col(BLUE, a), width=4)
    c.rect((1450, gy - 46, 1670, gy + 46), 20, GREEN, a, glow=True)
    c.text((1560, gy), "GPU compute", 30, BG0, a, 700, "mm")
    c.d.line((x0 + lw + 14, gy, 1450, gy), fill=col(GREEN, a), width=4)
    c.text((cx, y0 + n * (lh + gap) + 40), "28 layers · Llama 3.2 3B", 28, DIM, a, 500, "mm")


def s_overlap(c, t, dur):
    c.words(160, 190, "Reading and computing, at the same time.", 62, t)
    a = out3((t - 0.3) / 0.5)
    x0, wpx = 380, 1300
    scale = wpx / 0.5
    p = clamp((t - 0.6) / 2.6)
    # top: sequential (faded reference)
    y0 = 320
    c.text((x0, y0 - 18), "Load, then compute", 30, MUTED, a, 600, "ls")
    c.rect((x0, y0, x0 + scale * 0.30, y0 + 44), 10, BLUE, a * 0.45)
    c.rect((x0 + scale * 0.30, y0 + 56, x0 + scale * 0.46, y0 + 100), 10, GREEN, a * 0.45)
    c.text((x0 + scale * 0.46 + 20, y0 + 78), "0.46 s", 34, MUTED, a, 700, "lm")
    # bottom: streamed, 28 interleaved segments finishing at 0.29 s
    y1 = 560
    c.text((x0, y1 - 18), "Streamed", 30, FG, a, 700, "ls")
    n, total = 28, 0.29
    seg = total / n
    for i in range(n):
        s0 = i * seg
        if s0 / total <= p:
            c.rect((x0 + scale * s0 + 1, y1, x0 + scale * (s0 + seg * 0.92), y1 + 44), 5, BLUE, a, glow=True)
        g0 = s0 + seg * 0.35
        if g0 / total <= p:
            c.rect((x0 + scale * g0 + 1, y1 + 56, x0 + scale * (g0 + seg * 0.6), y1 + 100), 5, GREEN, a, glow=True)
    k = out3((t - 3.0) / 0.5)
    c.text((x0 + scale * 0.29 + 20, y1 + 50), "0.29 s", 46, GREEN, k, 800, "lm", glow=True)
    bx0, bx1 = x0 + scale * 0.29, x0 + scale * 0.46
    if k > 0:
        c.d.line((bx0, y1 + 140, bx1, y1 + 140), fill=col(AMBER, k), width=4)
        c.d.line((bx0, y1 + 125, bx0, y1 + 155), fill=col(AMBER, k), width=4)
        c.d.line((bx1, y1 + 125, bx1, y1 + 155), fill=col(AMBER, k), width=4)
        c.text(((bx0 + bx1) / 2, y1 + 190), "0.17 s saved", 36, AMBER, k, 700, "mm")


def node(c, x, y, w, h, title, sub, cc, a, glow=False):
    c.rect((x, y, x + w, y + h), 24, (24, 29, 40), a)
    c.rect((x, y, x + w, y + h), 24, cc, a, outline=cc, width=3)
    if glow:
        c.g.rounded_rectangle((x, y, x + w, y + h), 24, outline=col(cc, a), width=10)
    c.text((x + w / 2, y + h / 2 - 16), title, 40, FG, a, 700, "mm")
    c.text((x + w / 2, y + h / 2 + 28), sub, 26, MUTED, a, 500, "mm")


def s_unified(c, t, dur):
    c.words(W / 2, 170, "Apple Silicon: no copy step.", 66, t, anchor_center=True)
    # typical discrete-GPU path, struck through
    a = out3((t - 0.4) / 0.5)
    y = 300
    c.text((W / 2, y), "Discrete GPU:  SSD  →  CPU RAM  →  copy  →  GPU VRAM", 36, MUTED, a, 600, "mm")
    s = out3((t - 1.4) / 0.5)
    if s > 0:
        tw = c.d.textlength("Discrete GPU:  SSD  →  CPU RAM  →  copy  →  GPU VRAM", font=font(36, 600))
        c.d.line((W / 2 - tw / 2, y, W / 2 - tw / 2 + tw * s, y), fill=col(RED, 0.9), width=5)
    k = out3((t - 2.0) / 0.6)
    node(c, 240, 520, 360, 200, "SSD", "cache on disk", BLUE, k)
    node(c, 780, 500, 360, 240, "Unified memory", "CPU + GPU share it", VIOLET, k, glow=True)
    node(c, 1320, 520, 360, 200, "GPU", "runs the layer", GREEN, k)
    # moving packets
    if k > 0:
        for i in range(5):
            ph = ((t * 0.8 + i / 5) % 1)
            px = lerp(600, 780, ph)
            c.d.ellipse((px - 9, 611, px + 9, 629), fill=col(BLUE, k))
            c.g.ellipse((px - 14, 606, px + 14, 634), fill=col(BLUE, k))
            qx = lerp(1140, 1320, ph)
            c.d.ellipse((qx - 9, 611, qx + 9, 629), fill=col(GREEN, k))
            c.g.ellipse((qx - 14, 606, qx + 14, 634), fill=col(GREEN, k))
    k3 = out3((t - 3.6) / 0.5)
    c.text((W / 2, 850), "Bytes from the SSD land where the GPU reads them.", 38, FG, k3, 500, "mm")


def s_results(c, t, dur):
    c.words(160, 170, "Resuming a 6K-token chat on an M4 MacBook", 52, t, c=MUTED, w=600)
    a = out3((t - 0.5) / 0.5)
    c.text((160, 300), "BEFORE", 28, DIM, a, 700)
    c.text((160, 340), "0.46 s", 150, MUTED, a, 800)
    s = out3((t - 2.2) / 0.4)
    if s > 0:
        c.d.line((160, 435, 160 + 470 * s, 435), fill=col(RED, 0.85), width=8)
    k = out3((t - 2.8) / 0.6)
    val = lerp(0.46, 0.29, inout((t - 2.8) / 1.4))
    c.text((900, 300), "STREAMED", 28, GREEN, k, 700)
    c.text((900, 340), f"{val:.2f} s", 150, GREEN, k, 800, glow=True)
    b = back((t - 4.6) / 0.6)
    if b > 0:
        cx, cy = 1180, 640
        wpx, hpx = 420 * b, 120 * b
        c.rect((cx - wpx / 2, cy - hpx / 2, cx + wpx / 2, cy + hpx / 2), 60 * b, GREEN, 1, glow=True)
        if b > 0.6:
            c.text((cx, cy), "1.6× faster", int(64 * b), BG0, 1, 800, "mm")
    k2 = out3((t - 6.2) / 0.5)
    c.rect((160, 600, 640, 680), 40, GREEN, k2, outline=GREEN, width=3)
    c.text((400, 640), "✓  bit-identical output", 34, GREEN, k2, 700, "mm")
    k3 = out3((t - 7.4) / 0.5)
    c.text((160, 790), "256-token message:  1.25 s → 0.99 s  (1.3×)", 40, FG, k3, 600)
    c.text((160, 900), "Llama 3.2 3B · 4-bit · cold SSD read · median of 6 resumes", 28, DIM, k3, 500)


def s_sweet(c, t, dur):
    c.words(W / 2, 200, "Short messages are the sweet spot.", 70, t, anchor_center=True)
    msgs = [("tool_result: 3 tests passed", BLUE), ("“looks good, ship it”", GREEN),
            ("tool_result: file saved", BLUE), ("“now add docs”", GREEN)]
    for i, (m, cc) in enumerate(msgs):
        k = out3((t - 0.7 - i * 0.45) / 0.5)
        x = 420 + (i % 2) * 260
        y = 340 + i * 120
        c.rect((x + (1 - k) * 60, y, x + 820 + (1 - k) * 60, y + 92), 24, (26, 31, 42), k)
        c.rect((x + (1 - k) * 60, y, x + 12 + (1 - k) * 60, y + 92), 6, cc, k, glow=True)
        c.text((x + 40 + (1 - k) * 60, y + 46), m, 34, FG, k, 500, "lm")
    k = out3((t - 2.8) / 0.5)
    c.text((W / 2, 900), "…exactly what AI agents send.", 46, AMBER, k, 700, "mm", glow=True)


def s_cta(c, t, dur):
    k = back((t - 0.1) / 0.8)
    if k > 0:
        c.text((W / 2, 400), "purgekv", int(150 * (0.8 + 0.2 * k)), FG, clamp(k), 800, "mm", glow=True)
    a = out3((t - 0.8) / 0.5)
    c.text((W / 2, 520), "Layer-streamed KV resume for MLX", 48, MUTED, a, 600, "mm")
    chips = ["Open source · MIT", "Benchmarks + raw data", "Link below"]
    x = W / 2 - 3 * 190 - 40
    for i, s in enumerate(chips):
        b = out3((t - 1.4 - i * 0.25) / 0.5)
        wpx = 360
        cx = W / 2 + (i - 1) * 420
        c.rect((cx - wpx / 2, 640, cx + wpx / 2, 720), 40, (30, 36, 50), b)
        c.text((cx, 680), s, 32, GREEN if i == 2 else FG, b, 600, "mm")


SCENES = [  # vo index, min duration, tint, fn, transition into this scene
    (1, 6.6, AMBER, s_hook, None),
    (2, 7.6, BLUE, s_sequential, "push"),
    (3, 7.0, BLUE, s_layers, "zoom"),
    (4, 5.6, GREEN, s_overlap, "wipe"),
    (5, 7.6, VIOLET, s_unified, "push_up"),
    (6, 11.6, GREEN, s_results, "zoom"),
    (7, 5.8, AMBER, s_sweet, "wipe"),
    (8, 6.0, GREEN, s_cta, "zoom"),
]
LEAD, TAIL, TR = 0.45, 0.9, 0.6


def read_audio(path):
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def timeline():
    vos, starts, durs, t = [], [], [], 0.0
    for vi, mind, *_ in SCENES:
        v = read_audio(os.path.join(VO_DIR, f"vo{vi}.aiff"))
        d = max(mind, LEAD + len(v) / SR + TAIL)
        vos.append(v); starts.append(t); durs.append(d)
        t += d - TR  # next scene overlaps by the transition
    return vos, starts, durs, starts[-1] + durs[-1]


# ---------------- transitions ----------------

def transition(kind, a_img, b_img, p):
    p = inout(p)
    if kind == "push":
        out = Image.new("RGB", (W, H))
        out.paste(a_img, (int(-W * p), 0)); out.paste(b_img, (int(W * (1 - p)), 0))
        return out
    if kind == "push_up":
        out = Image.new("RGB", (W, H))
        out.paste(a_img, (0, int(-H * p))); out.paste(b_img, (0, int(H * (1 - p))))
        return out
    if kind == "zoom":
        s = 1 + 0.12 * p
        za = a_img.resize((int(W * s), int(H * s)), Image.BILINEAR).crop(
            (int((W * s - W) / 2), int((H * s - H) / 2), int((W * s - W) / 2) + W, int((H * s - H) / 2) + H))
        sb = 0.92 + 0.08 * p
        zb = Image.new("RGB", (W, H), BG0)
        bb = b_img.resize((int(W * sb), int(H * sb)), Image.BILINEAR)
        zb.paste(bb, (int((W - W * sb) / 2), int((H - H * sb) / 2)))
        return Image.blend(za, zb, p)
    if kind == "wipe":  # expanding circle reveal
        mask = Image.new("L", (W, H), 0)
        r = p * math.hypot(W, H) / 2 * 1.05
        ImageDraw.Draw(mask).ellipse((W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r), fill=255)
        return Image.composite(b_img, a_img, mask.filter(ImageFilter.GaussianBlur(20)))
    return Image.blend(a_img, b_img, p)


def scene_frame(i, t):
    vi, mind, tint, fn, _ = SCENES[i]
    c = Canvas()
    fn(c, t, DURS[i])
    return c.finish(background(STARTS[i] + t, tint))


# ---------------- audio ----------------

def synth_music(total):
    n = int(total * SR)
    t = np.arange(n) / SR
    bpm = 96
    beat = 60 / bpm
    bar = beat * 4
    # Am – F – C – G, two bars each
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    pad = np.zeros(n, np.float32)
    seg = bar * 2
    for ci in range(int(total / seg) + 1):
        ch = chords[ci % 4]
        s0, s1 = int(ci * seg * SR), min(n, int((ci + 1) * seg * SR) + int(0.4 * SR))
        if s0 >= n:
            break
        tt = t[s0:s1] - ci * seg
        env = np.minimum(1, tt / 0.8) * np.minimum(1, np.maximum(0, (seg + 0.4 - tt) / 0.8))
        for m in ch:
            for det in (-0.08, 0.08):
                f = hz(m) * 2 ** (det / 12)
                pad[s0:s1] += env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.12 * np.sin(6 * np.pi * f * tt))
        bass_f = hz(ch[0] - 12)
        pad[s0:s1] += 0.9 * env * np.sin(2 * np.pi * bass_f * tt)
    pad *= 0.06 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t))
    # arpeggio plucks on 8th notes
    arp = np.zeros(n, np.float32)
    step = beat / 2
    for k in range(int(total / step)):
        ch = chords[int(k * step / seg) % 4]
        note = ch[[0, 1, 2, 1][k % 4]] + 12
        s0 = int(k * step * SR); L = int(0.5 * SR)
        if s0 + L > n:
            break
        tt = np.arange(L) / SR
        arp[s0:s0 + L] += np.sin(2 * np.pi * hz(note) * tt) * np.exp(-tt * 7)
    arp *= 0.035
    # soft kick on beats, hats on off-beats, entering after the hook
    drums = np.zeros(n, np.float32)
    for k in range(int(total / beat)):
        s0 = int(k * beat * SR)
        if s0 / SR < 6.0 or s0 + int(0.4 * SR) > n:
            continue
        tt = np.arange(int(0.35 * SR)) / SR
        drums[s0:s0 + len(tt)] += 0.22 * np.sin(2 * np.pi * (45 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
        h0 = s0 + int(beat / 2 * SR)
        hl = int(0.05 * SR)
        if h0 + hl < n:
            nz = np.random.default_rng(k).standard_normal(hl).astype(np.float32)
            nz = np.diff(nz, prepend=0)  # crude high-pass
            drums[h0:h0 + hl] += 0.025 * nz * np.exp(-np.arange(hl) / SR * 80)
    music = pad + arp + drums
    fade = np.minimum(1, t / 1.5) * np.minimum(1, (total - t) / 2.5)
    return (music * fade).astype(np.float32)


def whoosh(dur=0.7, rng=0):
    L = int(dur * SR)
    nz = np.random.default_rng(rng).standard_normal(L).astype(np.float32)
    out = np.zeros(L, np.float32)
    # sweep a moving-average low-pass from dark to bright and back
    cs = np.cumsum(np.concatenate([[0], nz]))
    for i0 in range(0, L, 512):
        p = i0 / L
        win = int(lerp(60, 4, math.sin(math.pi * p)))
        i1 = min(L, i0 + 512)
        idx = np.arange(i0, i1)
        lo = np.maximum(0, idx - win)
        out[i0:i1] = (cs[idx + 1] - cs[lo + 1]) / np.maximum(1, idx - lo)
    env = np.sin(np.pi * np.linspace(0, 1, L)) ** 2
    return 0.5 * out * env / (np.abs(out).max() + 1e-6)


def impact():
    L = int(1.0 * SR); tt = np.arange(L) / SR
    boom = np.sin(2 * np.pi * (38 + 50 * np.exp(-tt * 12)) * tt) * np.exp(-tt * 4)
    shimmer = sum(np.sin(2 * np.pi * f * tt) for f in (880, 1320, 1760)) * np.exp(-tt * 3) * 0.08
    return (0.6 * boom + shimmer).astype(np.float32)


def tick():
    L = int(0.04 * SR); tt = np.arange(L) / SR
    return (0.15 * np.sin(2 * np.pi * 2200 * tt) * np.exp(-tt * 120)).astype(np.float32)


def build_audio(vos, starts, durs, total):
    n = int(total * SR) + SR
    voice = np.zeros(n, np.float32)
    for v, s in zip(vos, starts):
        i = int((s + LEAD) * SR); voice[i:i + len(v)] += v * 0.95
    music = synth_music(n / SR)
    # duck music under the voice
    env = np.convolve(np.abs(voice), np.ones(int(0.25 * SR)) / (0.25 * SR), "same")
    duck = 1 - 0.55 * np.clip(env / 0.02, 0, 1)
    sfx = np.zeros(n, np.float32)
    for k, s in enumerate(starts[1:], 1):
        w = whoosh(rng=k); i = int((s) * SR); sfx[i:i + len(w)] += w * 0.35
    ri = SCENES.index(next(x for x in SCENES if x[3] is s_results))
    rs = starts[ri]
    for tt in np.arange(2.8, 4.2, 0.09):  # counter ticks while the number counts down
        i = int((rs + tt) * SR); tk = tick(); sfx[i:i + len(tk)] += tk
    im = impact(); i = int((rs + 4.6) * SR); sfx[i:i + len(im)] += im * 0.7
    im2 = impact(); i = int((starts[-1] + 0.1) * SR); sfx[i:i + len(im2)] += im2 * 0.5
    if NO_VOICE:  # music-only cut: no narration, no ducking, music brought up front
        mix = music * 1.75 + sfx * 0.8
    else:
        mix = voice + music * duck * 1.2 + sfx
    mix = np.tanh(mix * 1.1) * 0.9
    st = np.stack([mix, mix], 1)
    return (st * 32767).astype(np.int16)


# ---------------- render ----------------
VOS, STARTS, DURS, TOTAL = timeline()


def frame_at(T):
    # active scene(s) at absolute time T
    for i in range(len(SCENES)):
        s, d = STARTS[i], DURS[i]
        if s <= T < s + d:
            nxt = i + 1
            if nxt < len(SCENES) and T >= STARTS[nxt]:  # inside a transition
                p = (T - STARTS[nxt]) / TR
                return transition(SCENES[nxt][4], scene_frame(i, T - s), scene_frame(nxt, T - STARTS[nxt]), p)
            return scene_frame(i, T - s)
    return scene_frame(len(SCENES) - 1, DURS[-1] - 0.01)


def main():
    nframes = int(TOTAL * FPS)
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        for i in range(len(SCENES)):
            frame_at(STARTS[i] + DURS[i] * 0.8).save(os.path.join(PREVIEW, f"v2_scene{i + 1}.png"))
        frame_at(STARTS[1] + TR * 0.5).save(os.path.join(PREVIEW, "v2_transition_push.png"))
        frame_at(STARTS[3] + TR * 0.5).save(os.path.join(PREVIEW, "v2_transition_wipe.png"))
    if ONLY_PREVIEW:
        print("timeline", [round(s, 1) for s in STARTS], round(TOTAL, 1)); return
    wav = OUT.replace(".mp4", ".wav")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(build_audio(VOS, STARTS, DURS, TOTAL)[: int(TOTAL * SR)].tobytes())
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT],
                         stdin=subprocess.PIPE)
    for f in range(nframes):
        p.stdin.write(frame_at(f / FPS).tobytes())
        if f % 300 == 0:
            print(f"frame {f}/{nframes}", flush=True)
    p.stdin.close(); p.wait()
    os.remove(wav)
    print(OUT, f"{TOTAL:.1f}s")


if __name__ == "__main__":
    main()
