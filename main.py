"""
Regression Quest — two-player linear regression exploration game.

Players take turns clicking in the (w0, w1) weight space to find the best
linear fit for a mystery dataset. Dots are colored red (hot/low error) to
blue (cold/high error). After each guess: WARMER or COLDER vs your last pick.
10 turns total, alternating. Lowest MSE wins.

Deploy with:
    pip install pygbag
    pygbag main.py
Then upload build/web/ to GitHub Pages.
"""

import asyncio, math, colorsys, random
import pygame
import numpy as np

# ── Layout ─────────────────────────────────────────────────────────────────────
SW, SH   = 960, 600
GS       = 480                          # grid size (square, pixels)
ML, MR   = 65, 15                       # left / right margin inside left panel
MT, MB   = 55, 65                       # top / bottom margin inside left panel
LP       = ML + GS + MR                 # left panel width  = 560
RP       = SW - LP                      # right panel width = 400

# ── Game settings ──────────────────────────────────────────────────────────────
WLO, WHI = -5.0, 5.0
WRANGE   = WHI - WLO
TURNS    = 10

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = (12,  12,  22)
GRIDBG = (18,  18,  32)
GRIDLN = (35,  35,  58)
AXCOL  = (70,  70, 115)
TXT    = (220, 220, 240)
DIM    = (110, 110, 148)
DIV    = (45,  45,  78)
WARM   = (255,  90,  60)
COLD   = ( 60, 120, 255)
PCOL   = [(80, 180, 255), (255, 140, 50)]   # P1 cyan-blue, P2 orange

# ── Coordinate helpers ─────────────────────────────────────────────────────────
def w2p(w0, w1):
    """Weight-space → screen pixel."""
    x = ML + (w0 - WLO) / WRANGE * GS
    y = MT + (WHI - w1) / WRANGE * GS
    return int(x), int(y)

def p2w(x, y):
    """Screen pixel → weight-space."""
    return (WLO + (x - ML) / GS * WRANGE,
            WHI - (y - MT) / GS * WRANGE)

def in_grid(x, y):
    return ML <= x <= ML + GS and MT <= y <= MT + GS

# ── MSE & colour ───────────────────────────────────────────────────────────────
def calc_mse(xs, ys, w1, w0):
    return float(np.mean((ys - (w1 * xs + w0)) ** 2))

def heat(val, lo, hi):
    """Map MSE to colour: blue (cold/high) → red (hot/low), log-scaled."""
    t = (math.log(val + 0.1) - lo) / (hi - lo + 1e-9)
    t = max(0.0, min(1.0, t))          # t=0 → hottest, t=1 → coldest
    r, g, b = colorsys.hsv_to_rgb(t * 0.67, 0.92, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

# ── Game class ─────────────────────────────────────────────────────────────────
class Game:
    def __init__(self):
        self.phase = "intro"
        self._gen_data()

    def _gen_data(self):
        tw1 = random.uniform(-2.5, 2.5)
        tw0 = random.uniform(-2.5, 2.5)
        xs  = np.random.uniform(-4.5, 4.5, 30)
        ys  = tw1 * xs + tw0 + np.random.normal(0, 0.9, 30)
        self.xs, self.ys = xs, ys
        # log-MSE range across the whole weight grid for colour normalisation
        vals = [calc_mse(xs, ys, v1, v0)
                for v0 in np.linspace(WLO, WHI, 20)
                for v1 in np.linspace(WLO, WHI, 20)]
        self.log_lo = math.log(min(vals) + 0.1)
        self.log_hi = math.log(max(vals) + 0.1)

    def start(self):
        self._gen_data()
        self.guesses   = []
        self.turn      = 0
        self.player    = 0
        self.last_mse  = [None, None]
        self.best      = [None, None]   # best guess per player
        self.fb_text   = ""
        self.fb_timer  = 0
        self.phase     = "play"

    def click(self, mx, my):
        if not in_grid(mx, my):
            return
        w0, w1 = p2w(mx, my)
        w0 = max(WLO, min(WHI, w0))
        w1 = max(WLO, min(WHI, w1))

        v  = calc_mse(self.xs, self.ys, w1, w0)
        p  = self.player
        lm = self.last_mse[p]

        if lm is not None:
            self.fb_text  = "WARMER!" if v < lm else "COLDER!"
            self.fb_timer = 150
        else:
            self.fb_text  = ""

        self.last_mse[p] = v

        if self.best[p] is None or v < self.best[p]["mse"]:
            self.best[p] = {"w0": w0, "w1": w1, "mse": v}

        self.guesses.append({
            "w0": w0, "w1": w1, "mse": v,
            "col": heat(v, self.log_lo, self.log_hi),
            "bdr": PCOL[p], "p": p,
        })

        self.turn += 1
        if self.turn >= TURNS:
            self.phase = "end"
        else:
            self.player = 1 - self.player

    def tick(self):
        if self.fb_timer > 0:
            self.fb_timer -= 1

# ── Draw helpers ───────────────────────────────────────────────────────────────
def _divider(s, y, x0=None, x1=None):
    x0 = x0 if x0 is not None else LP + 12
    x1 = x1 if x1 is not None else SW - 12
    pygame.draw.line(s, DIV, (x0, y), (x1, y))

# ── Left panel: weight space ───────────────────────────────────────────────────
def draw_weight_space(s, game, F):
    pygame.draw.rect(s, GRIDBG, (ML, MT, GS, GS))

    for v in range(-5, 6):
        ax, _  = w2p(v, 0)
        _,  ay = w2p(0, v)
        c = AXCOL if v == 0 else GRIDLN
        pygame.draw.line(s, c, (ax, MT),      (ax, MT + GS))
        pygame.draw.line(s, c, (ML, ay),      (ML + GS, ay))

    pygame.draw.rect(s, AXCOL, (ML, MT, GS, GS), 2)

    sf = F["sm"]
    for v in [-4, -2, 0, 2, 4]:
        ax, _  = w2p(v, 0)
        _,  ay = w2p(0, v)
        l = sf.render(str(v), True, DIM)
        s.blit(l, (ax - l.get_width() // 2,  MT + GS + 6))
        l = sf.render(str(v), True, DIM)
        s.blit(l, (ML - l.get_width() - 6,   ay - l.get_height() // 2))

    xl = F["nm"].render("w\u2080  (intercept)", True, TXT)
    s.blit(xl, (ML + GS // 2 - xl.get_width() // 2, MT + GS + 30))
    yl = pygame.transform.rotate(F["nm"].render("w\u2081  (slope)", True, TXT), 90)
    s.blit(yl, (4, MT + GS // 2 - yl.get_height() // 2))

    tt = F["ti"].render("WEIGHT SPACE", True, TXT)
    s.blit(tt, (LP // 2 - tt.get_width() // 2, 16))

    for i, g in enumerate(game.guesses):
        px, py = w2p(g["w0"], g["w1"])
        pygame.draw.circle(s, g["col"], (px, py), 8)
        pygame.draw.circle(s, g["bdr"], (px, py), 8, 2)
        # extra ring on the very last dot
        if i == len(game.guesses) - 1:
            pygame.draw.circle(s, g["bdr"], (px, py), 13, 2)

# ── Left panel: scatter plot (end screen) ─────────────────────────────────────
def draw_scatter(s, game, F):
    xs, ys = game.xs, game.ys
    xmin, xmax = -5.0, 5.0
    mg   = 0.8
    ymin = float(np.min(ys)) - mg
    ymax = float(np.max(ys)) + mg

    def d2p(x, y):
        px = ML + (x - xmin) / (xmax - xmin) * GS
        py = MT + (ymax - y) / (ymax - ymin) * GS
        return int(px), int(py)

    pygame.draw.rect(s, GRIDBG, (ML, MT, GS, GS))
    pygame.draw.rect(s, AXCOL,  (ML, MT, GS, GS), 2)

    for x, y in zip(xs, ys):
        pygame.draw.circle(s, TXT, d2p(x, y), 4)

    # clip regression lines to the plot rectangle
    s.set_clip(pygame.Rect(ML, MT, GS, GS))
    for pi, b in enumerate(game.best):
        if b:
            p1 = d2p(xmin, b["w1"] * xmin + b["w0"])
            p2 = d2p(xmax, b["w1"] * xmax + b["w0"])
            pygame.draw.line(s, PCOL[pi], p1, p2, 3)
    s.set_clip(None)

    # player legend on the plot
    sf = F["sm"]
    for pi, label in enumerate(["P1 best", "P2 best"]):
        dot_x = ML + 14 + pi * 90
        dot_y = MT + GS - 14
        if game.best[pi]:
            pygame.draw.line(s, PCOL[pi], (dot_x, dot_y), (dot_x + 30, dot_y), 3)
            l = sf.render(label, True, PCOL[pi])
            s.blit(l, (dot_x + 34, dot_y - l.get_height() // 2))

    tt = F["ti"].render("BEST FITS", True, TXT)
    s.blit(tt, (LP // 2 - tt.get_width() // 2, 16))
    xl = F["nm"].render("x", True, TXT)
    s.blit(xl, (ML + GS // 2, MT + GS + 30))
    yl = pygame.transform.rotate(F["nm"].render("y", True, TXT), 90)
    s.blit(yl, (4, MT + GS // 2 - yl.get_height() // 2))

# ── Right panel: playing ───────────────────────────────────────────────────────
def draw_right_play(s, game, F):
    nf, tf, bf, sf = F["nm"], F["ti"], F["bg"], F["sm"]
    x, y = LP + 22, 28
    pygame.draw.line(s, DIV, (LP, 0), (LP, SH), 2)

    s.blit(tf.render(f"Turn  {game.turn + 1} / {TURNS}", True, TXT), (x, y));  y += 50
    s.blit(bf.render(f"Player {game.player + 1}", True, PCOL[game.player]), (x, y)); y += 44
    s.blit(nf.render("Click the weight space", True, DIM), (x, y)); y += 24
    s.blit(nf.render("to place your guess!", True, DIM), (x, y)); y += 52

    if game.fb_text and game.fb_timer > 0:
        fc = WARM if game.fb_text == "WARMER!" else COLD
        s.blit(bf.render(game.fb_text, True, fc), (x, y))
    y += 52

    _divider(s, y); y += 20
    s.blit(nf.render("Best so far:", True, DIM), (x, y)); y += 30
    for pi in range(2):
        b   = game.best[pi]
        val = f"{b['mse']:.2f}" if b else "\u2014"
        s.blit(nf.render(f"Player {pi + 1}:  MSE = {val}", True, PCOL[pi]), (x, y)); y += 28

    y += 15; _divider(s, y); y += 15
    s.blit(sf.render("History:", True, DIM), (x, y)); y += 22

    for g in reversed(game.guesses[-8:]):
        pn   = "P1" if g["p"] == 0 else "P2"
        line = sf.render(
            f"{pn}  w\u2080={g['w0']:+.1f}  w\u2081={g['w1']:+.1f}  MSE={g['mse']:.2f}",
            True, PCOL[g["p"]]
        )
        s.blit(line, (x, y)); y += 20
        if y > SH - 15:
            break

# ── Right panel: end ──────────────────────────────────────────────────────────
def draw_right_end(s, game, F):
    nf, tf, bf, sf = F["nm"], F["ti"], F["bg"], F["sm"]
    x, y = LP + 22, 28
    pygame.draw.line(s, DIV, (LP, 0), (LP, SH), 2)

    s.blit(tf.render("RESULTS", True, TXT), (x, y)); y += 50

    b0, b1 = game.best[0], game.best[1]
    if b0 and b1:
        if   b0["mse"] < b1["mse"]: winner, wc = "Player 1 wins!", PCOL[0]
        elif b1["mse"] < b0["mse"]: winner, wc = "Player 2 wins!", PCOL[1]
        else:                        winner, wc = "It's a tie!",    TXT
    else:
        winner, wc = "\u2014", TXT

    s.blit(bf.render(winner, True, wc), (x, y)); y += 55

    _divider(s, y); y += 20
    for pi in range(2):
        b = game.best[pi]
        if b:
            s.blit(nf.render(f"Player {pi + 1}:", True, PCOL[pi]), (x, y)); y += 26
            s.blit(sf.render(f"  w\u2080 = {b['w0']:+.2f}", True, PCOL[pi]), (x, y)); y += 20
            s.blit(sf.render(f"  w\u2081 = {b['w1']:+.2f}", True, PCOL[pi]), (x, y)); y += 20
            s.blit(sf.render(f"  MSE = {b['mse']:.3f}", True, PCOL[pi]), (x, y)); y += 35
        else:
            s.blit(nf.render(f"Player {pi + 1}: no guesses", True, PCOL[pi]), (x, y)); y += 40

    _divider(s, y); y += 20
    s.blit(nf.render("Press SPACE or click to play again", True, DIM), (x, y))

# ── Intro screen ───────────────────────────────────────────────────────────────
def draw_intro(s, F):
    nf, tf, bf, sf = F["nm"], F["ti"], F["bg"], F["sm"]
    y = 65

    tt = bf.render("REGRESSION QUEST", True, TXT)
    s.blit(tt, (SW // 2 - tt.get_width() // 2, y)); y += 62

    rows = [
        (nf, "Two players take turns exploring the weight space.", TXT),
        (nf, "Each click tests a slope w\u2081 and intercept w\u2080 pair.", TXT),
        (sf, "", DIM),
        (nf, "Dot colour shows how good your guess is:", TXT),
        (nf, "  RED   = low error   (HOT \u2014 you're close!)",  WARM),
        (nf, "  BLUE  = high error  (COLD \u2014 keep searching)", COLD),
        (sf, "", DIM),
        (nf, "After each guess you'll see WARMER or COLDER", TXT),
        (nf, "compared to your own previous guess.", DIM),
        (sf, "", DIM),
        (nf, "10 turns total, alternating between players.", TXT),
        (nf, "Lowest MSE at the end wins!", TXT),
        (sf, "", DIM),
        (bf, "Press  SPACE  or click to start", (155, 220, 155)),
    ]
    for font, text, color in rows:
        if text:
            t = font.render(text, True, color)
            s.blit(t, (SW // 2 - t.get_width() // 2, y))
        y += font.get_height() + 6

# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Regression Quest")
    clock = pygame.time.Clock()

    F = {
        "bg": pygame.font.SysFont("Arial", 30, bold=True),
        "ti": pygame.font.SysFont("Arial", 22, bold=True),
        "nm": pygame.font.SysFont("Arial", 18),
        "sm": pygame.font.SysFont("Arial", 14),
    }

    game = Game()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if game.phase in ("intro", "end"):
                    game.start()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if game.phase in ("intro", "end"):
                    game.start()
                elif game.phase == "play":
                    game.click(mx, my)

        game.tick()
        screen.fill(BG)

        if game.phase == "intro":
            draw_intro(screen, F)
        elif game.phase == "play":
            draw_weight_space(screen, game, F)
            draw_right_play(screen, game, F)
        else:  # end
            draw_scatter(screen, game, F)
            draw_right_end(screen, game, F)

        pygame.display.flip()
        await asyncio.sleep(0)     # yield to browser event loop (Pygbag)
        clock.tick(60)

asyncio.run(main())
