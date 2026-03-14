"""
car_game.py  –  Self-Driving Car with realistic F1-style circuit & environment.

Drop-in replacement for game.py.  Interface:
    reset()            → initial state array  (NUM_SENSORS floats in [0,1])
    play_step(action)  → (reward, game_over, score)
    get_state()        → numpy float32 array, shape (NUM_SENSORS,)
"""

import pygame
import math
import random
import numpy as np

# ── Screen ───────────────────────────────────────────────────────────────────
SCREEN_W = 1100
SCREEN_H = 750

# ── Car physics ───────────────────────────────────────────────────────────────
CAR_SPEED  = 4.0    # px / frame  (constant forward speed)
TURN_SPEED = 5.0    # degrees / frame
TIMEOUT    = 2000   # max frames per episode

# ── Sensors ───────────────────────────────────────────────────────────────────
NUM_SENSORS    = 9
SENSOR_MAX     = 220
# Denser in the forward arc so the car sees corners earlier
SENSOR_OFFSETS = [-90, -60, -30, -15, 0, 15, 30, 60, 90]

# ── Checkpoints (evenly spaced around circuit, car must hit in order) ─────────
# Subset of TRACK_PTS indices chosen to cover every major section
CHECKPOINTS = [
    (700, 660),   # CP1 — end of main straight
    (970, 540),   # CP2 — T1 exit
    (900, 200),   # CP3 — top braking zone
    (760, 100),   # CP4 — hairpin apex
    (560, 240),   # CP5 — S-curves
    (320, 280),   # CP6 — chicane
    (160, 500),   # CP7 — slow left apex
    (220, 620),   # CP8 — return chicane
]
CHECKPOINT_RADIUS  = 55   # px — how close the car must get to collect
CHECKPOINT_REWARD  = 50   # reward per checkpoint
LAP_REWARD         = 200  # bonus for completing a full lap

# Total state size: 7 sensors + angle-to-next-CP + dist-to-next-CP
STATE_SIZE = NUM_SENSORS + 2

# ── Track ─────────────────────────────────────────────────────────────────────
TRACK_WIDTH = 95   # road width in pixels (wider = more margin for learning)

# Closed-loop centerline waypoints (first == last to close the loop)
TRACK_PTS = [
    (140, 660),   # START / FINISH
    (300, 660),   # main straight
    (500, 660),   # main straight
    (700, 660),   # main straight
    (880, 660),   # braking zone
    (950, 610),   # T1 apex
    (970, 540),   # T1 exit
    (950, 440),   # back straight
    (930, 320),   # back straight
    (900, 200),   # top braking zone
    (840, 130),   # hairpin entry
    (760, 100),   # hairpin apex
    (680, 120),   # hairpin exit
    (620, 180),   # S-curve entry
    (560, 240),   # S-curve mid
    (490, 200),   # S-curve right kink
    (400, 220),   # chicane entry
    (320, 280),   # chicane left
    (260, 340),   # chicane right
    (200, 420),   # slow left-hander entry
    (160, 500),   # slow left apex
    (180, 570),   # slow left exit
    (220, 620),   # return chicane
    (180, 650),   # return left flick
    (140, 660),   # START / FINISH (loop closed)
]

# ── Colour palette ────────────────────────────────────────────────────────────
C_GRASS_A    = ( 48, 122,  48)
C_GRASS_B    = ( 40, 105,  40)
C_ROAD       = ( 62,  62,  67)
C_BORDER     = (228, 228, 228)
C_DASH       = (255, 215,  45)
C_SENSOR     = (255, 215,   0)
C_HUD        = (255, 255, 255)
C_TREE_SH    = ( 18,  72,  18)
C_TREE_OUT   = ( 28,  98,  28)
C_TREE_MID   = ( 42, 130,  42)
C_TREE_HI    = ( 62, 165,  62)
C_TRUNK      = ( 92,  58,  22)
C_STAND_BODY = (148, 152, 162)
C_STAND_ROOF = (185,  38,  38)
C_CAR_BODY   = (205,  28,  28)   # red livery
C_CAR_WING   = ( 22,  22,  22)
C_CAR_COCK   = ( 12,  12,  12)
C_CAR_WHEEL  = ( 18,  18,  18)
C_CAR_RIM    = ( 75,  75,  75)
C_CAR_STRIPE = (255, 205,   0)


# ── Standalone draw helpers (called during track build & per-frame) ───────────

def _draw_tree(surf: pygame.Surface, cx: int, cy: int, r: int = 14) -> None:
    """Top-down tree: dark shadow + layered canopy + highlight spot."""
    pygame.draw.circle(surf, C_TREE_SH,  (cx + 3, cy + 3), r)
    pygame.draw.circle(surf, C_TREE_OUT, (cx,     cy),     r)
    pygame.draw.circle(surf, C_TREE_MID, (cx,     cy),     int(r * 0.68))
    pygame.draw.circle(surf, C_TREE_HI,  (cx - 3, cy - 3), int(r * 0.32))


def _draw_person(surf: pygame.Surface, cx: int, cy: int,
                 shirt: tuple) -> None:
    """Tiny top-down spectator: shirt disc + skin head."""
    pygame.draw.circle(surf, shirt,             (cx, cy),     4)
    pygame.draw.circle(surf, (238, 192, 135),   (cx, cy - 6), 3)


def _draw_grandstand(surf: pygame.Surface,
                     x: int, y: int, w: int, h: int) -> None:
    """Concrete grandstand with roof and three rows of coloured seats."""
    pygame.draw.rect(surf, C_STAND_BODY, (x, y, w, h))
    pygame.draw.rect(surf, C_STAND_ROOF, (x, y, w, 9))       # roof band
    seat_colors = [(210, 28, 28), (38, 78, 210), (248, 208, 25)]
    row_h = max(1, (h - 12) // 3)
    for row in range(3):
        for col in range(w // 10):
            sc = seat_colors[(row + col) % 3]
            pygame.draw.rect(surf, sc,
                             (x + col * 10 + 1, y + 11 + row * row_h, 8, row_h - 1))


def _draw_advert(surf: pygame.Surface,
                 x: int, y: int, w: int, h: int,
                 bg: tuple, fg: tuple = (255, 255, 255), text: str = "") -> None:
    """Trackside advertising hoarding."""
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, fg, (x, y, w, h), 2,  border_radius=3)
    if text:
        try:
            fnt = pygame.font.SysFont("arial", max(9, h - 6), bold=True)
            lbl = fnt.render(text, True, fg)
            surf.blit(lbl, lbl.get_rect(center=(x + w // 2, y + h // 2)))
        except Exception:
            pass


def _draw_flag_line(surf: pygame.Surface,
                    x: int, y1: int, y2: int) -> None:
    """Chequered start/finish line."""
    sq = 8
    rows = (y2 - y1) // sq
    for row in range(rows):
        for col in range(2):
            c = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
            pygame.draw.rect(surf, c, (x + col * sq, y1 + row * sq, sq, sq))


def _build_road_layer(surf: pygame.Surface, pts: list,
                      width: int, color: tuple) -> None:
    """Draw a rounded thick-polyline (road/border layer)."""
    r = width // 2
    n = len(pts)
    for i in range(n - 1):
        pygame.draw.line(surf, color, pts[i], pts[i + 1], width)
        pygame.draw.circle(surf, color, pts[i], r)
    pygame.draw.circle(surf, color, pts[-1], r)


# ── Main environment class ────────────────────────────────────────────────────

class CarGameAI:

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self, w: int = SCREEN_W, h: int = SCREEN_H):
        self.w, self.h = w, h
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Self-Driving Car — DQN")
        self.clock = pygame.time.Clock()
        self.hud_font = pygame.font.SysFont("arial", 22, bold=True)

        self._build_track()   # visual surface + collision mask
        self.reset()

    # ── Track & environment builder ───────────────────────────────────────────
    def _build_track(self) -> None:
        rng = random.Random(7)   # fixed seed → deterministic layout

        # ── Visual surface ────────────────────────────────────────────────────
        vis = pygame.Surface((self.w, self.h))

        # 1) Grass (subtle two-tone checkerboard)
        vis.fill(C_GRASS_A)
        for bx in range(0, self.w, 44):
            for by in range(0, self.h, 44):
                if (bx // 44 + by // 44) % 2 == 0:
                    pygame.draw.rect(vis, C_GRASS_B, (bx, by, 44, 44))

        # 2) Grandstands (drawn before road so road covers any overlap)
        _draw_grandstand(vis, 195, 712, 510, 52)   # main-straight stand
        _draw_grandstand(vis,  15, 450,  50, 90)   # left-sector stand
        _draw_grandstand(vis, 955, 458,  80, 60)   # T1 stand

        # 3) Outer trees
        outer_trees = [
            # top-left cluster
            (55, 55), (95, 80), (65, 125), (125, 62), (145, 108),
            (50, 168), (88, 192),
            # bottom-left
            (55, 710), (82, 728), (52, 738), (110, 720),
            # right side
            (1025, 118), (1062, 162), (1042, 238), (1055, 318),
            (1030, 408), (1068, 458), (1048, 528), (1038, 608),
            (985, 668), (1018, 698),
            # top strip
            (320,  42), (400,  56), (480,  38), (560,  55),
            (640,  40), (720,  60), (800,  45),
        ]
        for tx, ty in outer_trees:
            _draw_tree(vis, tx, ty, rng.randint(12, 20))

        # 4) Road border — wider than road to leave a visible kerb zone each side
        _build_road_layer(vis, TRACK_PTS, TRACK_WIDTH + 22, C_BORDER)

        # 5) Road tarmac
        _build_road_layer(vis, TRACK_PTS, TRACK_WIDTH, C_ROAD)

        # 6) Start / finish line + pole-position box
        sl_x = 240
        _draw_flag_line(vis, sl_x - 8, 625, 700)
        pygame.draw.rect(vis, (255, 215, 0), (sl_x - 8, 618, 16, 8), 2)  # pole marker

        # 8) Centre-line dashes on main straight
        for dash_x in range(300, 860, 28):
            pygame.draw.line(vis, C_DASH,
                             (dash_x, 659), (dash_x + 14, 659), 2)

        # 9) Advertising boards at apexes
        _draw_advert(vis, 940, 620, 60, 22, (25, 115, 200), text="GP")
        _draw_advert(vis, 955, 435, 52, 20, (160, 30, 160), text="T2")
        _draw_advert(vis, 808,  82, 60, 20, (200,  45,  25), text="APEX")
        _draw_advert(vis, 230, 295, 55, 20, (20, 155,  80), text="CHICANE")
        _draw_advert(vis,  68, 482, 48, 20, (200, 140,  20), text="S3")

        # 10) Circuit title on tarmac (infield area)
        try:
            title_font = pygame.font.SysFont("arial", 17, bold=True)
            title = title_font.render("FORMULA AI GRAND PRIX", True, (175, 175, 185))
            vis.blit(title, (390, 540))
        except Exception:
            pass

        # 11) Infield trees (drawn on top of road border so they appear inside)
        infield_trees = [
            (570, 420), (610, 465), (555, 488), (640, 505),
            (680, 395), (705, 448), (735, 478), (762, 428),
            (500, 448), (478, 492), (542, 538),
            (405, 398), (442, 442), (382, 458),
            (655, 348), (692, 318), (732, 358),
            (520, 360), (478, 380), (440, 360),
        ]
        for tx, ty in infield_trees:
            _draw_tree(vis, tx, ty, rng.randint(9, 15))

        # 12) Spectators
        shirts = [
            (215, 38, 38), (38, 78, 215), (255, 208, 28),
            (48, 175, 75), (195, 195, 195), (255, 135, 18),
        ]
        # Main grandstand spectators
        for px in range(210, 700, 11):
            for py in range(718, 758, 13):
                _draw_person(vis, px, py, rng.choice(shirts))
        # T1 grandstand spectators
        for px in range(960, 1030, 12):
            for py in range(465, 512, 13):
                _draw_person(vis, px, py, rng.choice(shirts))
        # Left stand spectators
        for px in range(20, 62, 12):
            for py in range(458, 538, 13):
                _draw_person(vis, px, py, rng.choice(shirts))

        self.track_surf = vis

        # ── Collision mask (pure road shape, no decor) ────────────────────────
        mask = pygame.Surface((self.w, self.h))
        mask.fill((0, 0, 0))
        _build_road_layer(mask, TRACK_PTS, TRACK_WIDTH, (255, 255, 255))
        # Cache as numpy array: shape (w, h, 3), index as [x, y]
        self._road_pixels = pygame.surfarray.array3d(mask)

    # ── Road check (uses collision mask) ──────────────────────────────────────
    def _on_road(self, x: float, y: float) -> bool:
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.w and 0 <= yi < self.h):
            return False
        return bool(self._road_pixels[xi, yi, 0] > 128)   # white = road

    # ── Episode management ────────────────────────────────────────────────────
    def reset(self):
        # Start on main straight, just after start line, facing east
        self.x      = 280.0
        self.y      = 660.0
        self.angle  = 0.0     # 0=east, 90=south (pygame y-down), etc.

        self.score        = 0
        self.frame_iter   = 0
        self.sensor_dists = [float(SENSOR_MAX)] * NUM_SENSORS
        self.next_cp      = 0   # index into CHECKPOINTS
        self.laps         = 0
        self.prev_x       = self.x   # for progress calculation
        self.prev_y       = self.y
        return self.get_state()

    # ── Core step ─────────────────────────────────────────────────────────────
    def play_step(self, action):
        """
        Args:
            action: [straight, right, left]  (one-hot list)
        Returns:
            (reward, game_over, score)
        """
        self.frame_iter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Steering
        if action[1] == 1:
            self.angle = (self.angle + TURN_SPEED) % 360
        elif action[2] == 1:
            self.angle = (self.angle - TURN_SPEED) % 360

        # Move forward
        rad = math.radians(self.angle)
        self.x += CAR_SPEED * math.cos(rad)
        self.y += CAR_SPEED * math.sin(rad)

        # Sensors
        self.sensor_dists = self._cast_sensors()

        # Off-track crash
        if not self._on_road(self.x, self.y):
            self._update_ui()
            return -10, True, self.score

        # Timeout
        if self.frame_iter >= TIMEOUT:
            self._update_ui()
            return -1, True, self.score

        # ── Reward shaping ─────────────────────────────────────────────────
        cx, cy = CHECKPOINTS[self.next_cp]

        # 1) Dense progress: reward how much closer we got to the next CP this frame
        prev_dist = math.hypot(self.prev_x - cx, self.prev_y - cy)
        curr_dist = math.hypot(self.x       - cx, self.y       - cy)
        progress  = (prev_dist - curr_dist) / CAR_SPEED   # 1.0 = heading straight at CP
        reward = 1.0 + max(progress, 0.0) * 2.0           # range [1, 3] per frame

        # 2) Wall proximity penalty — discourages hugging walls / zigzagging
        min_sensor = min(self.sensor_dists) if self.sensor_dists else SENSOR_MAX
        if min_sensor < 30:
            reward -= (30.0 - min_sensor) / 30.0 * 1.5    # up to -1.5 near wall

        # 3) Checkpoint gate
        if curr_dist < CHECKPOINT_RADIUS:
            reward += CHECKPOINT_REWARD
            self.next_cp = (self.next_cp + 1) % len(CHECKPOINTS)
            if self.next_cp == 0:
                reward += LAP_REWARD
                self.laps += 1

        self.prev_x = self.x
        self.prev_y = self.y
        self.score += 1
        self._update_ui()
        self.clock.tick(60)
        return reward, False, self.score

    # ── State ─────────────────────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        # 7 normalised sensor distances
        dists = self._cast_sensors()
        state = [d / SENSOR_MAX for d in dists]

        # Angle to next checkpoint relative to car heading  (-1 = hard left, +1 = hard right)
        cx, cy = CHECKPOINTS[self.next_cp]
        abs_angle = math.degrees(math.atan2(cy - self.y, cx - self.x))
        rel_angle  = (abs_angle - self.angle + 180) % 360 - 180   # -180 … +180
        state.append(rel_angle / 180.0)

        # Normalised distance to next checkpoint (0 = here, 1 = far away)
        dist_to_cp = math.hypot(cx - self.x, cy - self.y)
        state.append(min(dist_to_cp / 600.0, 1.0))

        return np.array(state, dtype=np.float32)   # shape: (STATE_SIZE,) = (9,)

    # ── Sensors ───────────────────────────────────────────────────────────────
    def _cast_sensors(self):
        return [self._ray(self.angle + off) for off in SENSOR_OFFSETS]

    def _ray(self, deg: float) -> float:
        rad     = math.radians(deg)
        cos_a   = math.cos(rad)
        sin_a   = math.sin(rad)
        for d in range(1, SENSOR_MAX + 1):
            if not self._on_road(self.x + cos_a * d, self.y + sin_a * d):
                return float(d)
        return float(SENSOR_MAX)

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _update_ui(self) -> None:
        # Track environment
        self.display.blit(self.track_surf, (0, 0))

        # Sensor beams
        for i, off in enumerate(SENSOR_OFFSETS):
            rad  = math.radians(self.angle + off)
            dist = self.sensor_dists[i]
            ex   = int(self.x + math.cos(rad) * dist)
            ey   = int(self.y + math.sin(rad) * dist)
            # Beam line (fades from bright to dim)
            pygame.draw.line(self.display, C_SENSOR,
                             (int(self.x), int(self.y)), (ex, ey), 1)
            # Endpoint dot
            pygame.draw.circle(self.display, (255, 80, 80), (ex, ey), 3)

        # ── F1-style top-down car ──────────────────────────────────────────
        car_w, car_h = 44, 22
        car_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)

        # Rear wing (wide horizontal bar)
        pygame.draw.rect(car_surf, C_CAR_WING, (0, 0, 8, car_h))

        # Main body (tapered trapezoid)
        body = [(8, 5), (36, 3), (car_w, car_h // 2), (36, car_h - 3), (8, car_h - 5)]
        pygame.draw.polygon(car_surf, C_CAR_BODY, body)

        # Nose cone gradient effect
        nose = [(34, 6), (car_w, car_h // 2), (34, car_h - 6)]
        pygame.draw.polygon(car_surf, (175, 20, 20), nose)

        # Front wing
        pygame.draw.rect(car_surf, C_CAR_WING, (36, 0, 8, car_h))

        # Cockpit / halo arch
        pygame.draw.ellipse(car_surf, C_CAR_COCK, (15, 7, 14, 8))

        # Gold team stripe along body
        pygame.draw.line(car_surf, C_CAR_STRIPE, (8, car_h // 2), (34, car_h // 2), 2)

        # Four wheels
        for wx, wy in [(31, 2), (31, car_h - 2), (11, 2), (11, car_h - 2)]:
            pygame.draw.circle(car_surf, C_CAR_WHEEL, (wx, wy), 4)
            pygame.draw.circle(car_surf, C_CAR_RIM,   (wx, wy), 2)

        # Rotate and blit
        rotated = pygame.transform.rotate(car_surf, -self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        self.display.blit(rotated, rect)

        # ── Next checkpoint marker (green pulsing ring) ───────────────────
        cx, cy = CHECKPOINTS[self.next_cp]
        pygame.draw.circle(self.display, (0, 230, 80),  (int(cx), int(cy)), CHECKPOINT_RADIUS, 2)
        pygame.draw.circle(self.display, (0, 255, 100), (int(cx), int(cy)), 6)

        # ── HUD ───────────────────────────────────────────────────────────
        panel = pygame.Surface((240, 72), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.display.blit(panel, (8, 8))
        self.display.blit(self.hud_font.render(f"Score : {self.score}",  True, C_HUD), (14, 12))
        self.display.blit(self.hud_font.render(f"Laps  : {self.laps}",   True, C_HUD), (14, 34))
        self.display.blit(self.hud_font.render(
            f"CP    : {self.next_cp + 1}/{len(CHECKPOINTS)}", True, (0, 230, 80)), (14, 56))

        pygame.display.flip()
