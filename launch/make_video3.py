"""Launch video for purgekv 0.2.0: large chats, prefetch, shared prompts. Music only.

  python launch/make_video3.py OUT.mp4 [--preview DIR] [--frames-only]
Reuses the look, transitions and synthesized music of launch/make_video2.py.
Numbers are from results/scale_Llama-3.2-3B-Instruct-4bit.json and
results/scale_Llama-3.2-1B-Instruct-4bit.json (Apple M4, 16 GB, cold SSD reads).
"""
import os, subprocess, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video2 as v
from make_video2 import W, H, FPS, SR, FG, MUTED, DIM, LINE, BLUE, GREEN, RED, AMBER, VIOLET, BG0
from make_video2 import clamp, out3, inout, back, lerp, col, chat_bubble

CARD = (26, 31, 42)


# ---------------- scenes ----------------

def s_hook(c, t, dur):
    k = out3(t / 0.5)
    c.rect((W / 2 - 150, 120, W / 2 + 150, 180), 30, VIOLET, k * 0.9, glow=True)
    c.text((W / 2, 150), "PART 2", 30, BG0, k, 800, "mm")
    c.words(W / 2, 300, "What about a really long chat?", 80, t - 0.4, anchor_center=True)
    k2 = out3((t - 1.4) / 0.5)
    chat_bubble(c, 460, 400 + (1 - k2) * 30, 1000, 110, "Here's the whole codebase. Find the bug.", k2, (36, 44, 60))
    k3 = out3((t - 2.2) / 0.5)
    if k3 > 0:
        n = int(lerp(0, 32768, inout((t - 2.2) / 2.2)))
        c.text((W / 2, 680), f"{n:,}", 170, AMBER, k3, 800, "mm", glow=True)
        c.text((W / 2, 800), "tokens of history to bring back", 40, MUTED, k3, 600, "mm")


def s_recompute(c, t, dur):
    c.words(160, 180, "Re-reading 32K tokens from scratch:", 60, t, c=FG)
    a = out3((t - 0.6) / 0.5)
    x0, y0, full = 160, 300, 1600
    c.text((x0, y0), "Recompute (no cache)", 34, MUTED, a, 600)
    p = clamp((t - 0.9) / 3.2)
    c.rect((x0, y0 + 50, x0 + full, y0 + 130), 20, LINE, a)
    c.rect((x0, y0 + 50, x0 + full * p, y0 + 130), 20, RED, a, glow=True)
    c.text((x0 + full - 20, y0 + 90), f"{169 * p:.0f} s", 44, FG, a, 800, "rm")
    b = out3((t - 3.6) / 0.5)
    c.text((x0, y0 + 260), "Reload saved KV cache from SSD", 34, MUTED, b, 600)
    c.rect((x0, y0 + 310, x0 + max(28, full * 1.80 / 169), y0 + 390), 14, AMBER, b, glow=True)
    c.text((x0 + 60, y0 + 350), "1.80 s", 44, AMBER, b, 800, "lm")
    k = out3((t - 4.8) / 0.5)
    c.text((W / 2, 900), "Saving the cache already wins big. Can we get rid of the 1.8 s too?", 40, FG, k, 600, "mm")


def s_prefetch(c, t, dur):
    c.words(W / 2, 170, "Idea: read while you type.", 76, t, anchor_center=True)
    x0, x1 = 240, 1680
    lane_y = [380, 520, 660]
    labels = [("You", GREEN), ("SSD", AMBER), ("GPU", BLUE)]
    a = out3((t - 0.5) / 0.5)
    for (lab, cc), y in zip(labels, lane_y):
        c.text((x0 - 40, y + 40), lab, 36, cc, a, 700, "rm")
        c.rect((x0, y, x1, y + 80), 16, (20, 24, 34), a)
    p = clamp((t - 1.0) / 3.4)  # timeline playhead
    enter = 0.72
    # You: typing until Enter
    tw = min(p, enter)
    c.rect((x0, lane_y[0], x0 + (x1 - x0) * tw, lane_y[0] + 80), 16, GREEN, a * 0.85)
    if p > 0.05:
        c.text((x0 + 24, lane_y[0] + 40), "typing a message…", 30, BG0, a, 700, "lm")
    # SSD: prefetch runs during typing
    sw = min(p, 0.5)
    c.rect((x0, lane_y[1], x0 + (x1 - x0) * sw, lane_y[1] + 80), 16, AMBER, a * 0.85, glow=True)
    if p > 0.05:
        c.text((x0 + 24, lane_y[1] + 40), "prefetch the chat's KV", 30, BG0, a, 700, "lm")
    if p > 0.5:
        c.text((x0 + (x1 - x0) * 0.5 + 24, lane_y[1] + 40), "done, waiting in memory", 30, AMBER, a, 600, "lm")
    # GPU: only compute after Enter
    if p > enter:
        gw = min(p, 0.86) - enter
        c.rect((x0 + (x1 - x0) * enter, lane_y[2], x0 + (x1 - x0) * (enter + gw), lane_y[2] + 80), 16, BLUE, a, glow=True)
    xe = x0 + (x1 - x0) * enter
    if p > enter:
        k = out3((p - enter) / 0.05)
        c.d.line((xe, 340, xe, 770), fill=col(FG, k * 0.8), width=3)
        c.text((xe, 320), "⏎ Enter", 30, FG, k, 700, "mm")
    k = out3((t - 4.6) / 0.5)
    c.text((W / 2, 900), "When you hit Enter, the GPU only has your new message left to do.", 40, FG, k, 600, "mm")


def s_results(c, t, dur):
    c.words(160, 150, "Resume time, Llama 3.2 3B on a 16 GB M4 MacBook", 50, t, c=MUTED, w=600)
    rows = [("8K", 0.54, 0.32, 0.15), ("16K", 0.96, 0.63, 0.21), ("32K", 1.80, 1.40, 0.31)]
    names = [("Today", RED), ("Streamed", AMBER), ("Prefetched", GREEN)]
    a = out3((t - 0.4) / 0.5)
    for j, (nm, cc) in enumerate(names):
        c.rect((160 + j * 330, 210, 190 + j * 330, 240), 6, cc, a)
        c.text((205 + j * 330, 225), nm, 30, FG, a, 600, "lm")
    scale = 1150 / 1.80
    for i, row in enumerate(rows):
        y = 320 + i * 190
        k = out3((t - 0.8 - i * 0.5) / 0.5)
        c.text((330, y + 70), row[0], 52, FG, k, 800, "rm")
        for j, (val, (nm, cc)) in enumerate(zip(row[1:], names)):
            g = inout((t - 1.0 - i * 0.5 - j * 0.25) / 0.9)
            yy = y + j * 50
            c.rect((370, yy + 10, 370 + scale * val * g, yy + 50), 10, cc, k, glow=(j == 2))
            if g > 0.9:
                c.text((390 + scale * val, yy + 30), f"{val:.2f} s", 30, cc, k, 700, "lm")
    b = back((t - 4.4) / 0.6)
    if b > 0:
        cx, cy = 1560, 935
        wpx, hpx = 440 * b, 130 * b
        c.rect((cx - wpx / 2, cy - hpx / 2, cx + wpx / 2, cy + hpx / 2), 65 * b, GREEN, 1, glow=True)
        if b > 0.6:
            c.text((cx, cy), "5.8× faster", int(66 * b), BG0, 1, 800, "mm")
    k2 = out3((t - 5.4) / 0.5)
    c.text((160, 975), "32-token message · cold SSD read · median of 3 · ✓ same output every time", 30, DIM, k2, 500)


def s_bigger(c, t, dur):
    c.words(W / 2, 200, "Even longer: 64K tokens (Llama 3.2 1B)", 64, t, anchor_center=True)
    a = out3((t - 0.6) / 0.5)
    c.text((440, 380), "TODAY", 28, DIM, a, 700, "mm")
    c.text((440, 480), "1.06 s", 130, MUTED, a, 800, "mm")
    s = out3((t - 1.8) / 0.4)
    if s > 0:
        c.d.line((260, 485, 260 + 360 * s, 485), fill=col(RED, 0.85), width=8)
    k = out3((t - 2.2) / 0.6)
    val = lerp(1.06, 0.19, inout((t - 2.2) / 1.2))
    c.text((1360, 380), "PREFETCHED", 28, GREEN, k, 700, "mm")
    c.text((1360, 480), f"{val:.2f} s", 130, GREEN, k, 800, "mm", glow=True)
    k2 = out3((t - 3.6) / 0.5)
    c.rect((W / 2 - 520, 650, W / 2 + 520, 760), 30, CARD, k2)
    c.text((W / 2, 705), "Recomputing it instead:  205 s", 44, RED, k2, 700, "mm")


def s_memory(c, t, dur):
    c.words(W / 2, 190, "New reader: one copy, not two", 72, t, anchor_center=True)
    a = out3((t - 0.6) / 0.5)
    for j, (lab, val, cc) in enumerate((("Before", 7.1, RED), ("v0.2", 5.7, GREEN))):
        x = 560 + j * 520
        g = inout((t - 0.9 - j * 0.4) / 1.0)
        hpx = 480 * val / 7.1 * g
        c.rect((x, 820 - hpx, x + 280, 820), 18, cc, a, glow=(j == 1))
        c.text((x + 140, 860), lab, 36, FG, a, 700, "mm")
        if g > 0.9:
            c.text((x + 140, 790 - hpx), f"{val} GB", 54, cc, a, 800, "mm")
    k = out3((t - 2.8) / 0.5)
    c.text((W / 2, 960), "Peak memory at 32K tokens: −20%.  Loads: 21% faster.", 40, FG, k, 600, "mm")


def s_shared(c, t, dur):
    c.words(W / 2, 170, "Many chats, one cached prompt", 72, t, anchor_center=True)
    a = out3((t - 0.5) / 0.5)
    px, py = 260, 420
    c.rect((px, py, px + 420, py + 200), 26, VIOLET, a * 0.9, glow=True)
    c.text((px + 210, py + 75), "32K-token doc", 40, BG0, a, 800, "mm")
    c.text((px + 210, py + 130), "saved once", 32, BG0, a, 600, "mm")
    for i in range(4):
        k = out3((t - 1.0 - i * 0.3) / 0.5)
        y = 260 + i * 140
        x = 1000
        if k > 0:
            c.d.line((px + 420, py + 100, x, y + 50), fill=col(VIOLET, k * 0.6), width=4)
        c.rect((x + (1 - k) * 60, y, x + 520 + (1 - k) * 60, y + 100), 22, CARD, k)
        c.text((x + 40 + (1 - k) * 60, y + 50), f"Chat {i + 1}  ·  +256 own tokens", 34, FG, k, 600, "lm")
    k = out3((t - 2.8) / 0.5)
    c.text((W / 2, 930), "Disk:  15.2 GB  →  3.9 GB", 60, GREEN, k, 800, "mm", glow=True)


def s_cta(c, t, dur):
    k = back((t - 0.1) / 0.8)
    if k > 0:
        c.text((W / 2, 380), "purgekv 0.2.0", int(130 * (0.8 + 0.2 * k)), FG, clamp(k), 800, "mm", glow=True)
    a = out3((t - 0.8) / 0.5)
    c.text((W / 2, 500), "Fast resume for long local AI chats on Apple Silicon", 46, MUTED, a, 600, "mm")
    chips = ["Prefetch · 5.8×", "Shared prompts", "Open source · MIT"]
    for i, s in enumerate(chips):
        b = out3((t - 1.4 - i * 0.25) / 0.5)
        cx = W / 2 + (i - 1) * 440
        c.rect((cx - 190, 620, cx + 190, 700), 40, (30, 36, 50), b)
        c.text((cx, 660), s, 32, GREEN if i == 0 else FG, b, 600, "mm")
    k2 = out3((t - 2.4) / 0.5)
    c.text((W / 2, 830), "github.com/PoneeshKumar/purgekv", 44, GREEN, k2, 700, "mm", glow=True)


SCENES = [  # vo index (unused), duration, tint, fn, transition into this scene
    (0, 5.6, AMBER, s_hook, None),
    (0, 6.4, RED, s_recompute, "push"),
    (0, 6.6, BLUE, s_prefetch, "zoom"),
    (0, 7.4, GREEN, s_results, "wipe"),
    (0, 5.4, GREEN, s_bigger, "push_up"),
    (0, 5.0, VIOLET, s_memory, "zoom"),
    (0, 5.2, VIOLET, s_shared, "push"),
    (0, 5.6, GREEN, s_cta, "zoom"),
]
TR = v.TR


def timeline():
    starts, durs, t = [], [], 0.0
    for _, d, *_ in SCENES:
        starts.append(t); durs.append(d); t += d - TR
    return starts, durs, starts[-1] + durs[-1]


def build_audio(starts, total):
    n = int(total * SR) + SR
    music = v.synth_music(n / SR)
    sfx = np.zeros(n, np.float32)

    def add(clip, at, gain):
        i = int(at * SR); clip = clip[: max(0, n - i)]; sfx[i:i + len(clip)] += clip * gain
    for k, s in enumerate(starts[1:], 1):
        add(v.whoosh(rng=k), s, 0.35)
    for tt in np.arange(2.2, 4.4, 0.09):  # token counter in the hook
        add(v.tick(), tt, 0.8)
    add(v.impact(), starts[3] + 4.4, 0.7)  # 5.8× badge
    for tt in np.arange(2.2, 3.4, 0.09):  # 1.06 -> 0.19 countdown
        add(v.tick(), starts[4] + tt, 1.0)
    add(v.impact(), starts[6] + 2.8, 0.45)  # disk saving
    add(v.impact(), starts[-1] + 0.1, 0.5)
    mix = np.tanh((music * 1.75 + sfx * 0.8) * 1.1) * 0.9
    return (np.stack([mix, mix], 1) * 32767).astype(np.int16)


def main(out, preview=None, only_preview=False):
    starts, durs, total = timeline()
    v.SCENES, v.STARTS, v.DURS, v.TOTAL = SCENES, starts, durs, total
    if preview:
        os.makedirs(preview, exist_ok=True)
        for i in range(len(SCENES)):
            v.frame_at(starts[i] + durs[i] * 0.85).save(os.path.join(preview, f"v3_scene{i + 1}.png"))
    if only_preview:
        print("timeline", [round(s, 1) for s in starts], round(total, 1)); return
    wav = out.replace(".mp4", ".wav")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(build_audio(starts, total)[: int(total * SR)].tobytes())
    p = subprocess.Popen([v.FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    nframes = int(total * FPS)
    for f in range(nframes):
        p.stdin.write(v.frame_at(f / FPS).tobytes())
        if f % 300 == 0:
            print(f"frame {f}/{nframes}", flush=True)
    p.stdin.close(); p.wait()
    os.remove(wav)
    print(out, f"{total:.1f}s")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[a.index("--preview") + 1] if "--preview" in a else None, "--frames-only" in a)
