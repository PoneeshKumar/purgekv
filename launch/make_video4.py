"""Launch video for purgekv 0.3.0: purgekv-server for agents. Music only.

  python launch/make_video4.py OUT.mp4 [--preview DIR] [--frames-only]
Reuses the look, transitions and synthesized music of launch/make_video2.py.
Numbers: results/server_Llama-3.2-3B-Instruct-4bit.json (Exp K) and
results/trading_agents*/report.json (Exp L), Apple M4, 16 GB.
"""
import os, subprocess, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video2 as v
from make_video2 import W, H, FPS, SR, FG, MUTED, DIM, BLUE, GREEN, RED, AMBER, VIOLET, BG0
from make_video2 import clamp, out3, inout, back, lerp, col

CARD = (26, 31, 42)
AGENTS = [("Research", VIOLET), ("Engineer", BLUE), ("QA", AMBER), ("Backtest", GREEN), ("Execute", RED)]


def chip(c, cx, cy, label, cc, a, w=250, h=74, glow=False):
    c.rect((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), h / 2, CARD, a)
    c.rect((cx - w / 2, cy - h / 2, cx - w / 2 + 12, cy + h / 2), 6, cc, a, glow=glow)
    c.text((cx + 6, cy), label, 32, FG, a, 650, "mm")


# ---------------- scenes ----------------

def s_hook(c, t, dur):
    k = out3(t / 0.5)
    c.rect((W / 2 - 150, 120, W / 2 + 150, 180), 30, GREEN, k * 0.9, glow=True)
    c.text((W / 2, 150), "PART 3", 30, BG0, k, 800, "mm")
    c.words(W / 2, 300, "Five AI agents. One laptop.", 84, t - 0.4, anchor_center=True)
    for i, (name, cc) in enumerate(AGENTS):
        b = back((t - 1.4 - i * 0.18) / 0.5)
        if b > 0:
            chip(c, 300 + i * 330, 520, name, cc, clamp(b))
    k2 = out3((t - 2.8) / 0.5)
    c.text((W / 2, 700), "“Make and backtest a trading strategy.”", 46, MUTED, k2, 600, "mm")
    c.text((W / 2, 790), "Qwen3-8B · 16 GB M4 MacBook · all local", 36, DIM, k2, 600, "mm")


def s_problem(c, t, dur):
    c.words(W / 2, 170, "RAM fits about one agent's memory.", 66, t, anchor_center=True)
    a = out3((t - 0.6) / 0.5)
    box = (660, 300, 1260, 520)
    c.rect(box, 30, (20, 24, 34), a)
    c.text((960, 330), "RAM for chats: 1 GB", 30, MUTED, a, 600, "mm")
    order = [0, 1, 2, 3, 4, 0]
    p = clamp((t - 1.0) / 3.2) * (len(order) - 1)
    cur = int(p)
    name, cc = AGENTS[order[cur]]
    chip(c, 960, 440, name, cc, a, w=360, glow=True)
    for j in range(5):  # the others wait outside, evicted
        if j != order[cur]:
            nm, ccc = AGENTS[j]
            x = 260 + [0, 1, 2, 3, 4].index(j) * 350
            chip(c, x, 650, nm, ccc, a * 0.55, w=240)
    k = out3((t - 4.3) / 0.5)
    c.text((W / 2, 800), "Stock mlx_lm.server forgets the rest and re-reads every word:", 40, FG, k, 600, "mm")
    k2 = out3((t - 4.9) / 0.5)
    c.text((W / 2, 900), "29.3 s per agent turn", 76, RED, k2, 800, "mm", glow=True)


def s_how(c, t, dur):
    c.words(W / 2, 160, "purgekv-server: same API, plus an SSD tier", 62, t, anchor_center=True)
    a = out3((t - 0.5) / 0.5)
    ram, ssd = (300, 330, 820, 590), (1100, 330, 1620, 590)
    c.rect(ram, 30, (20, 24, 34), a); c.text((560, 370), "RAM", 36, BLUE, a, 800, "mm")
    c.rect(ssd, 30, (20, 24, 34), a); c.text((1360, 370), "SSD", 36, AMBER, a, 800, "mm")
    chip(c, 560, 470, "Engineer", BLUE, a, w=300, glow=True)
    # 1: after every reply, append to SSD
    k1 = out3((t - 1.0) / 0.5)
    p1 = clamp((t - 1.2) / 1.0)
    if k1 > 0:
        x = lerp(720, 1180, inout(p1))
        c.rect((x - 16, 450, x + 16, 490), 8, AMBER, k1, glow=True)
        c.text((W / 2, 650), "1  After every reply: its new tokens go to SSD", 34, AMBER, k1, 650, "mm")
    for i, (nm, cc) in enumerate(AGENTS[2:4]):
        chip(c, 1360, 450 + i * 90, nm, cc, k1 * 0.8, w=260)
    # 2: tool running -> prefetch back
    k2 = out3((t - 2.6) / 0.5)
    p2 = clamp((t - 2.8) / 1.0)
    if k2 > 0:
        x = lerp(1180, 720, inout(p2))
        c.rect((x - 16, 530, x + 16, 570), 8, GREEN, k2, glow=True)
        c.text((W / 2, 710), "2  While an agent runs a tool: SSD reads it back (prefetch)", 34, GREEN, k2, 650, "mm")
    k3 = out3((t - 4.0) / 0.5)
    c.text((W / 2, 830), "Its next turn finds its memory already loaded.", 44, FG, k3, 650, "mm")
    c.text((W / 2, 920), "3  --kv-bits 8: half the bytes to save and read", 36, MUTED, k3, 600, "mm")


def s_results(c, t, dur):
    c.words(160, 150, "4 coding agents, one server, 8K-token contexts each", 48, t, c=MUTED, w=600)
    rows = [("Stock mlx_lm.server", 29.3, RED), ("purgekv-server", 1.33, AMBER), ("+ 8-bit KV", 1.13, GREEN)]
    scale = 1150 / 29.3
    for i, (nm, val, cc) in enumerate(rows):
        y = 300 + i * 170
        k = out3((t - 0.6 - i * 0.4) / 0.5)
        c.text((160, y + 20), nm, 38, FG, k, 700, "lm")
        g = inout((t - 0.9 - i * 0.4) / 1.0)
        c.rect((160, y + 60, 160 + max(14, scale * val * g), y + 120), 14, cc, k, glow=(i == 2))
        if g > 0.9:
            c.text((190 + max(14, scale * val), y + 90), f"{val} s", 44, cc, k, 800, "lm")
    b = back((t - 3.4) / 0.6)
    if b > 0:
        cx, cy = 1500, 760
        c.rect((cx - 230 * b, cy - 65 * b, cx + 230 * b, cy + 65 * b), 65 * b, GREEN, 1, glow=True)
        if b > 0.6:
            c.text((cx, cy), "26× faster", int(66 * b), BG0, 1, 800, "mm")
    k2 = out3((t - 4.4) / 0.5)
    c.text((160, 960), "Llama 3.2 3B · time to first token, median · same replies as mlx-lm's own cache (20/20)", 28, DIM, k2, 500)


def s_restart(c, t, dur):
    c.words(W / 2, 200, "Restart the server?", 76, t, anchor_center=True)
    a = out3((t - 0.6) / 0.5)
    c.text((520, 380), "STOCK", 28, DIM, a, 700, "mm")
    c.text((520, 480), "45.6 s", 130, MUTED, a, 800, "mm")
    c.text((520, 580), "starts empty, recomputes", 30, DIM, a, 600, "mm")
    s = out3((t - 1.8) / 0.4)
    if s > 0:
        c.d.line((330, 485, 330 + 380 * s, 485), fill=col(RED, 0.85), width=8)
    k = out3((t - 2.2) / 0.6)
    val = lerp(45.6, 1.35, inout((t - 2.2) / 1.2))
    c.text((1400, 380), "PURGEKV", 28, GREEN, k, 700, "mm")
    c.text((1400, 480), f"{val:.2f} s", 130, GREEN, k, 800, "mm", glow=True)
    c.text((1400, 580), "reopens every chat from SSD", 30, GREEN, k, 600, "mm")


def s_real(c, t, dur):
    c.words(W / 2, 160, "The real task: 5 trading agents on Qwen3-8B", 60, t, anchor_center=True)
    for i, (name, cc) in enumerate(AGENTS):
        k = out3((t - 0.5 - i * 0.12) / 0.4)
        chip(c, 300 + i * 330, 280, name, cc, k, w=270, h=64)
    items = [("Average wait per agent turn", 15.7, 7.0, "2.2×"), ("Agents whose memory left RAM", 21.8, 6.9, "3.1×")]
    for i, (lab, a_, b_, ratio) in enumerate(items):
        y = 430 + i * 230
        k = out3((t - 1.4 - i * 0.8) / 0.5)
        c.text((200, y), lab, 38, FG, k, 650, "lm")
        c.text((200, y + 100), f"{a_} s", 80, MUTED, k, 800, "lm")
        c.text((560, y + 100), "→", 70, DIM, k, 700, "lm")
        g = out3((t - 1.9 - i * 0.8) / 0.5)
        c.text((680, y + 100), f"{b_} s", 80, GREEN, g, 800, "lm", glow=True)
        c.text((1080, y + 100), ratio, 64, GREEN, g, 800, "lm")
    k3 = out3((t - 3.6) / 0.5)
    c.text((W / 2, 980), "stock vs purgekv-server · 1 GB RAM for chats · paper trading only", 30, DIM, k3, 500, "mm")


def s_cta(c, t, dur):
    k = back((t - 0.1) / 0.8)
    if k > 0:
        c.text((W / 2, 330), "purgekv 0.3.0", int(130 * (0.8 + 0.2 * k)), FG, clamp(k), 800, "mm", glow=True)
    a = out3((t - 0.8) / 0.5)
    c.text((W / 2, 450), "A drop-in mlx_lm.server for agents on your Mac", 46, MUTED, a, 600, "mm")
    b = out3((t - 1.4) / 0.5)
    c.rect((W / 2 - 640, 550, W / 2 + 640, 650), 24, (16, 19, 27), b)
    c.text((W / 2 - 600, 600), "$ purgekv-server --model mlx-community/Qwen3-8B-4bit", 38, GREEN, b, 600, "lm")
    if b > 0.5 and int(t * 2) % 2 == 0:
        c.rect((W / 2 + 470, 578, W / 2 + 486, 622), 3, GREEN, b)
    k2 = out3((t - 2.4) / 0.5)
    c.text((W / 2, 780), "github.com/PoneeshKumar/purgekv", 46, GREEN, k2, 700, "mm", glow=True)
    c.text((W / 2, 860), "Open source · MIT", 34, MUTED, k2, 600, "mm")


SCENES = [  # vo index (unused), duration, tint, fn, transition into this scene
    (0, 5.4, GREEN, s_hook, None),
    (0, 7.0, RED, s_problem, "push"),
    (0, 6.6, AMBER, s_how, "zoom"),
    (0, 6.8, GREEN, s_results, "wipe"),
    (0, 5.0, BLUE, s_restart, "push_up"),
    (0, 6.4, VIOLET, s_real, "zoom"),
    (0, 6.0, GREEN, s_cta, "zoom"),
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
    for i in range(5):  # agent chips pop in
        add(v.tick(), 1.4 + i * 0.18, 1.2)
    add(v.impact(), starts[1] + 4.9, 0.4)       # 29.3 s
    add(v.impact(), starts[3] + 3.4, 0.7)       # 26x badge
    for tt in np.arange(2.2, 3.4, 0.09):        # 45.6 -> 1.35 countdown
        add(v.tick(), starts[4] + tt, 1.0)
    add(v.impact(), starts[-1] + 0.1, 0.5)
    mix = np.tanh((music * 1.75 + sfx * 0.8) * 1.1) * 0.9
    return (np.stack([mix, mix], 1) * 32767).astype(np.int16)


def main(out, preview=None, only_preview=False):
    starts, durs, total = timeline()
    v.SCENES, v.STARTS, v.DURS, v.TOTAL = SCENES, starts, durs, total
    if preview:
        os.makedirs(preview, exist_ok=True)
        for i in range(len(SCENES)):
            v.frame_at(starts[i] + durs[i] * 0.85).save(os.path.join(preview, f"v4_scene{i + 1}.png"))
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
    for f in range(int(total * FPS)):
        p.stdin.write(v.frame_at(f / FPS).tobytes())
    p.stdin.close(); p.wait()
    os.remove(wav)
    print(out, f"{total:.1f}s")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[a.index("--preview") + 1] if "--preview" in a else None, "--frames-only" in a)
