"""
warehouse_grid.py - interactive 2D warehouse sandbox (environment + visualiser)

Run:  python warehouse_grid.py

Team interfaces (keep these stable so 3 people can work in parallel):
    Warehouse.grid      numpy array [y, x] of FLOOR / SHELF / DROP / PICKUP
    Warehouse.robots    list[Robot]   (x, y, path, task, carrying)
    Warehouse.tasks     list[Task]    (pickup, drop, status)
    Warehouse.bfs(...)  PLACEHOLDER pathfinder  -> replace with A* + reservations
    Warehouse._auto_assign()  PLACEHOLDER greedy allocator -> replace with Hungarian

Controls
    Left-click / drag   paint with current tool (keys 1-4)
    Right-click robot   select it      Right-click cell   send selected robot there
    Space run/pause  D auto-dispatch  T spawn task  N/X add/remove robot at cursor
    R new layout  C clear  S/L save/load  G grid lines  +/- speed  Esc quit
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
COLS, ROWS = 22, 15
CELL, MARGIN, SIDEBAR = 36, 16, 280
FLOOR, SHELF, DROP, PICKUP = 0, 1, 2, 3
TOOLS = [("Shelf", SHELF), ("Erase", FLOOR), ("Drop station", DROP), ("Pickup point", PICKUP)]
SAVE_FILE = Path("warehouse_layout.json")

BG = (15, 17, 24)
PANEL = (24, 27, 38)
FLOOR_A, FLOOR_B = (33, 37, 51), (36, 41, 56)
GRID_LINE = (46, 52, 70)
SHELF_C, SHELF_TOP, SHELF_SH = (92, 108, 153), (128, 145, 195), (12, 14, 20)
DROP_C, PICK_C = (52, 211, 153), (251, 191, 36)
TEXT, MUTED, ACCENT = (226, 232, 240), (140, 150, 170), (129, 140, 248)
TOOL_COLORS = [SHELF_TOP, MUTED, DROP_C, PICK_C]
ROBOT_COLORS = [(239, 68, 68), (59, 130, 246), (168, 85, 247), (236, 72, 153),
                (20, 184, 166), (249, 115, 22), (132, 204, 22), (14, 165, 233)]


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
@dataclass
class Task:
    id: int
    pickup: tuple
    drop: tuple
    robot_id: int | None = None
    status: str = "pending"          # pending -> to_pickup -> to_drop -> (removed)


@dataclass
class Robot:
    id: int
    x: int
    y: int
    color: tuple
    path: list = field(default_factory=list)
    task: Task | None = None
    carrying: bool = False
    wait: int = 0
    rx: float = 0.0                  # smoothed render position
    ry: float = 0.0

    def __post_init__(self):
        self.rx, self.ry = float(self.x), float(self.y)


class Warehouse:
    def __init__(self, cols=COLS, rows=ROWS):
        self.cols, self.rows = cols, rows
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.robots: list[Robot] = []
        self.tasks: list[Task] = []
        self.tick = 0
        self.delivered = 0
        self._rid = 0
        self._tid = 0

    # ---- queries ----
    def in_bounds(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows

    def is_walkable(self, x, y):
        return self.in_bounds(x, y) and self.grid[y, x] != SHELF

    def neighbors(self, x, y):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if self.is_walkable(x + dx, y + dy):
                yield (x + dx, y + dy)

    def robot_at(self, x, y):
        return next((r for r in self.robots if r.x == x and r.y == y), None)

    def get_robot(self, rid):
        return next((r for r in self.robots if r.id == rid), None)

    # ---- layout ----
    def generate_layout(self, seed=None, n_robots=4):
        rng = random.Random(seed)
        self.grid[:] = FLOOR
        self.robots.clear()
        self.tasks.clear()
        self.tick = self.delivered = 0
        self._rid = self._tid = 0
        for x0 in range(2, self.cols - 2, 4):          # shelf blocks, 2-wide aisles
            for y0 in (2, 7):
                self.grid[y0:y0 + 4, x0:x0 + 2] = SHELF
        for x in np.linspace(3, self.cols - 4, 4).astype(int):   # drop stations
            self.grid[self.rows - 1, x] = DROP
        near_shelf = [(x, y) for y in range(self.rows) for x in range(self.cols)
                      if self.grid[y, x] == FLOOR
                      and any(self.grid[ny, nx] == SHELF for nx, ny in
                              ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                              if self.in_bounds(nx, ny))]
        for x, y in rng.sample(near_shelf, min(12, len(near_shelf))):
            self.grid[y, x] = PICKUP
        for i in range(n_robots):
            self.add_robot(1 + 2 * i, 0)

    def clear(self):
        self.grid[:] = FLOOR
        self.tasks.clear()
        for r in self.robots:
            r.task, r.path, r.carrying = None, [], False

    def validate_tasks(self):
        for t in list(self.tasks):
            if (self.grid[t.pickup[1], t.pickup[0]] != PICKUP
                    or self.grid[t.drop[1], t.drop[0]] != DROP):
                self.cancel_task(t)

    def cancel_task(self, t):
        for r in self.robots:
            if r.task is t:
                r.task, r.carrying, r.path = None, False, []
        if t in self.tasks:
            self.tasks.remove(t)

    # ---- robots / tasks ----
    def add_robot(self, x, y):
        if not self.is_walkable(x, y) or self.robot_at(x, y):
            return None
        r = Robot(self._rid, x, y, ROBOT_COLORS[self._rid % len(ROBOT_COLORS)])
        self._rid += 1
        self.robots.append(r)
        return r

    def remove_robot(self, r):
        if r.task:
            r.task.robot_id, r.task.status = None, "pending"
        self.robots.remove(r)

    def spawn_task(self):
        pickups = [(int(x), int(y)) for y, x in zip(*np.where(self.grid == PICKUP))]
        drops = [(int(x), int(y)) for y, x in zip(*np.where(self.grid == DROP))]
        if not pickups or not drops:
            return None
        used = {t.pickup for t in self.tasks}
        pool = [p for p in pickups if p not in used] or pickups
        t = Task(self._tid, random.choice(pool), random.choice(drops))
        self._tid += 1
        self.tasks.append(t)
        return t

    # ---- PLACEHOLDER pathfinding (replace with A* + reservation table) ----
    def bfs(self, start, goal, blocked=frozenset()):
        if start == goal:
            return []
        prev, q = {start: None}, deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nb in self.neighbors(*cur):
                if nb not in prev and (nb not in blocked or nb == goal):
                    prev[nb] = cur
                    q.append(nb)
        if goal not in prev:
            return None
        path, cur = [], goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    # ---- PLACEHOLDER allocation (replace with Hungarian) ----
    def _auto_assign(self):
        idle = [r for r in self.robots if r.task is None]
        for t in self.tasks:
            if t.robot_id is None and idle:
                r = min(idle, key=lambda r: abs(r.x - t.pickup[0]) + abs(r.y - t.pickup[1]))
                idle.remove(r)
                t.robot_id, t.status, r.task = r.id, "to_pickup", t
                r.path = self.bfs((r.x, r.y), t.pickup) or []

    def _advance_task(self, r):
        t, pos = r.task, (r.x, r.y)
        if t.status == "to_pickup" and pos == t.pickup:
            t.status, r.carrying = "to_drop", True
            r.path = self.bfs(pos, t.drop) or []
        elif t.status == "to_drop" and pos == t.drop:
            r.task, r.carrying = None, False
            self.tasks.remove(t)
            self.delivered += 1
        else:                                            # lost path -> replan
            goal = t.pickup if t.status == "to_pickup" else t.drop
            r.path = self.bfs(pos, goal) or []

    def step(self, auto=True):
        self.tick += 1
        if auto:
            self._auto_assign()
        for r in self.robots:
            pos = (r.x, r.y)
            if not r.path:
                if r.task:
                    self._advance_task(r)
                continue
            nxt = r.path[0]
            others = {(o.x, o.y) for o in self.robots if o is not r}
            if not self.is_walkable(*nxt):
                r.path = self.bfs(pos, r.path[-1], others) or []
                continue
            if nxt in others:                            # blocked -> wait, then replan
                r.wait += 1
                if r.wait >= 2:
                    r.path = self.bfs(pos, r.path[-1], others) or r.path
                    r.wait = 0
                continue
            r.wait = 0
            r.x, r.y = nxt
            r.path.pop(0)

    # ---- persistence ----
    def save(self, path=SAVE_FILE):
        data = {"grid": self.grid.tolist(), "robots": [[r.x, r.y] for r in self.robots]}
        Path(path).write_text(json.dumps(data))

    def load(self, path=SAVE_FILE):
        data = json.loads(Path(path).read_text())
        g = np.array(data["grid"], dtype=np.int8)
        if g.shape != self.grid.shape:
            raise ValueError("grid size mismatch")
        self.grid = g
        self.robots.clear()
        self.tasks.clear()
        self.tick = self.delivered = 0
        for x, y in data["robots"]:
            self.add_robot(x, y)


# ----------------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------------
class App:
    def __init__(self):
        pygame.init()
        self.gw, self.gh = COLS * CELL, ROWS * CELL
        self.W = 2 * MARGIN + self.gw + SIDEBAR
        self.H = max(2 * MARGIN + self.gh + 60, 680)
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Warehouse Sandbox")
        self.clock = pygame.time.Clock()
        name = "segoeui,helvetica,arial,dejavusans"
        self.f_lg = pygame.font.SysFont(name, 24, bold=True)
        self.f_md = pygame.font.SysFont(name, 16)
        self.f_sm = pygame.font.SysFont(name, 13, bold=True)
        self.f_id = pygame.font.SysFont(name, 15, bold=True)

        self.wh = Warehouse()
        self.wh.generate_layout(seed=7)
        self.tool = 0
        self.selected = None
        self.painting = False
        self.paused = False
        self.auto = True
        self.grid_lines = True
        self.step_ms, self.acc = 220, 0
        self.running = True
        self.toast_text, self.toast_until = "", 0
        for _ in range(3):
            self.wh.spawn_task()

    # ---- helpers ----
    def toast(self, msg):
        self.toast_text, self.toast_until = msg, pygame.time.get_ticks() + 2200

    def cell_rect(self, x, y):
        return pygame.Rect(MARGIN + x * CELL, MARGIN + y * CELL, CELL, CELL)

    def mouse_cell(self, pos):
        x, y = (pos[0] - MARGIN) // CELL, (pos[1] - MARGIN) // CELL
        return (x, y) if self.wh.in_bounds(x, y) else None

    def text(self, s, pos, font=None, color=TEXT, anchor="topleft"):
        surf = (font or self.f_md).render(s, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    # ---- input ----
    def handle_event(self, e):
        wh = self.wh
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            self.on_key(e.key)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            cell = self.mouse_cell(e.pos)
            if cell and e.button == 1:
                self.painting = True
                self.paint(cell)
            elif cell and e.button == 3:
                self.right_click(cell)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.painting = False
        elif e.type == pygame.MOUSEMOTION and self.painting:
            cell = self.mouse_cell(e.pos)
            if cell:
                self.paint(cell)

    def on_key(self, k):
        wh = self.wh
        cell = self.mouse_cell(pygame.mouse.get_pos())
        if k == pygame.K_ESCAPE:
            self.running = False
        elif pygame.K_1 <= k <= pygame.K_4:
            self.tool = k - pygame.K_1
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_d:
            self.auto = not self.auto
        elif k == pygame.K_t:
            if not wh.spawn_task():
                self.toast("Need at least one pickup and one drop")
        elif k == pygame.K_n and cell:
            if not wh.add_robot(*cell):
                self.toast("Can't place a robot there")
        elif k == pygame.K_x and cell:
            r = wh.robot_at(*cell)
            if r:
                wh.remove_robot(r)
                self.selected = None if self.selected == r.id else self.selected
        elif k == pygame.K_r:
            wh.generate_layout(seed=random.randrange(10_000))
            self.selected = None
            for _ in range(3):
                wh.spawn_task()
            self.toast("New layout")
        elif k == pygame.K_c:
            wh.clear()
            self.toast("Grid cleared")
        elif k == pygame.K_g:
            self.grid_lines = not self.grid_lines
        elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.step_ms = max(50, self.step_ms - 30)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.step_ms = min(600, self.step_ms + 30)
        elif k == pygame.K_s:
            wh.save()
            self.toast(f"Saved to {SAVE_FILE}")
        elif k == pygame.K_l:
            try:
                wh.load()
                self.selected = None
                self.toast("Layout loaded")
            except Exception as ex:
                self.toast(f"Load failed: {ex}")

    def paint(self, cell):
        x, y = cell
        value = TOOLS[self.tool][1]
        if value == SHELF and self.wh.robot_at(x, y):
            self.toast("A robot is standing there")
            return
        self.wh.grid[y, x] = value
        self.wh.validate_tasks()

    def right_click(self, cell):
        wh = self.wh
        r = wh.robot_at(*cell)
        if r:
            self.selected = r.id
            return
        sel = wh.get_robot(self.selected)
        if sel and wh.is_walkable(*cell):
            p = wh.bfs((sel.x, sel.y), cell)
            if p is None:
                self.toast("No route to that cell")
            else:
                sel.path = p

    # ---- update ----
    def update(self, dt):
        if not self.paused:
            self.acc += dt
            while self.acc >= self.step_ms:
                self.wh.step(self.auto)
                self.acc -= self.step_ms
        k = 1 - math.exp(-dt / 70)
        for r in self.wh.robots:
            r.rx += (r.x - r.rx) * k
            r.ry += (r.y - r.ry) * k

    # ---- drawing ----
    def draw(self):
        s, wh = self.screen, self.wh
        s.fill(BG)
        pygame.draw.rect(s, PANEL, (MARGIN - 6, MARGIN - 6, self.gw + 12, self.gh + 12), border_radius=14)
        active = {t.pickup for t in wh.tasks}
        for y in range(ROWS):
            for x in range(COLS):
                r = self.cell_rect(x, y)
                pygame.draw.rect(s, FLOOR_A if (x + y) % 2 == 0 else FLOOR_B, r)
                c = wh.grid[y, x]
                if c == SHELF:
                    self.draw_shelf(r)
                elif c == DROP:
                    self.draw_drop(r)
                elif c == PICKUP:
                    self.draw_pickup(r, (x, y) in active)
        if self.grid_lines:
            for x in range(COLS + 1):
                pygame.draw.line(s, GRID_LINE, (MARGIN + x * CELL, MARGIN), (MARGIN + x * CELL, MARGIN + self.gh))
            for y in range(ROWS + 1):
                pygame.draw.line(s, GRID_LINE, (MARGIN, MARGIN + y * CELL), (MARGIN + self.gw, MARGIN + y * CELL))
        self.draw_paths()
        for r in wh.robots:
            self.draw_robot(r, r.id == self.selected)
        cell = self.mouse_cell(pygame.mouse.get_pos())
        if cell:
            pygame.draw.rect(s, TOOL_COLORS[self.tool], self.cell_rect(*cell), 2, border_radius=6)
        self.draw_sidebar(cell)
        self.draw_footer(cell)
        pygame.display.flip()

    def draw_shelf(self, r):
        body = r.inflate(-6, -6)
        pygame.draw.rect(self.screen, SHELF_SH, body.move(2, 3), border_radius=6)
        pygame.draw.rect(self.screen, SHELF_C, body, border_radius=6)
        pygame.draw.rect(self.screen, SHELF_TOP, (body.x + 3, body.y + 3, body.w - 6, 4), border_radius=2)
        for dy in (14, 21):
            pygame.draw.line(self.screen, (70, 84, 125), (body.x + 5, body.y + dy), (body.right - 6, body.y + dy), 2)

    def draw_drop(self, r):
        b = r.inflate(-4, -4)
        pygame.draw.rect(self.screen, (24, 66, 58), b, border_radius=8)
        pygame.draw.rect(self.screen, DROP_C, b, 2, border_radius=8)
        cx, cy = r.center
        pygame.draw.polygon(self.screen, DROP_C, [(cx - 7, cy - 5), (cx + 7, cy - 5), (cx, cy + 7)])

    def draw_pickup(self, r, has_task):
        cx, cy = r.center
        pygame.draw.circle(self.screen, PICK_C, (cx, cy), 12, 2)
        if has_task:
            bx = pygame.Rect(0, 0, 14, 14)
            bx.center = (cx, cy)
            pygame.draw.rect(self.screen, PICK_C, bx, border_radius=3)
            pygame.draw.line(self.screen, (120, 80, 10), (cx, bx.top), (cx, bx.bottom), 2)
        else:
            pygame.draw.circle(self.screen, PICK_C, (cx, cy), 3)

    def draw_paths(self):
        ov = pygame.Surface((self.gw, self.gh), pygame.SRCALPHA)
        for r in self.wh.robots:
            if not r.path:
                continue
            pts = [(r.rx * CELL + CELL / 2, r.ry * CELL + CELL / 2)]
            pts += [(x * CELL + CELL / 2, y * CELL + CELL / 2) for x, y in r.path]
            sel = r.id == self.selected
            pygame.draw.lines(ov, (*r.color, 170 if sel else 95), False, pts, 4 if sel else 3)
            pygame.draw.circle(ov, (*r.color, 220), pts[-1], 7, 2)
        self.screen.blit(ov, (MARGIN, MARGIN))

    def draw_robot(self, r, selected):
        s = self.screen
        cx, cy = int(MARGIN + r.rx * CELL + CELL / 2), int(MARGIN + r.ry * CELL + CELL / 2)
        pygame.draw.ellipse(s, (8, 9, 13), (cx - 13, cy + 8, 26, 9))
        body = pygame.Rect(0, 0, 26, 26)
        body.center = (cx, cy)
        pygame.draw.rect(s, r.color, body, border_radius=8)
        pygame.draw.rect(s, tuple(min(255, c + 60) for c in r.color), body, 2, border_radius=8)
        self.text(str(r.id), (cx, cy), self.f_id, (255, 255, 255), "center")
        if r.carrying:
            pk = pygame.Rect(0, 0, 12, 12)
            pk.center = (cx, cy - 17)
            pygame.draw.rect(s, PICK_C, pk, border_radius=3)
            pygame.draw.rect(s, (120, 80, 10), pk, 2, border_radius=3)
        if selected:
            pygame.draw.circle(s, (255, 255, 255), (cx, cy), int(19 + 2 * math.sin(pygame.time.get_ticks() / 180)), 2)

    def draw_sidebar(self, cell):
        s, wh = self.screen, self.wh
        px = 2 * MARGIN + self.gw
        panel = pygame.Rect(px, MARGIN - 6, SIDEBAR - 10, self.H - 2 * (MARGIN - 6))
        pygame.draw.rect(s, PANEL, panel, border_radius=14)
        x0, x1 = px + 18, panel.right - 18
        self.text("Warehouse Sandbox", (x0, 24), self.f_lg)
        self.text("multi-robot grid environment", (x0, 54), self.f_md, MUTED)

        for i, (label, on) in enumerate([("PAUSED" if self.paused else "RUNNING", not self.paused),
                                         ("AUTO ON" if self.auto else "AUTO OFF", self.auto)]):
            pill = pygame.Rect(x0 + i * 106, 88, 98, 26)
            col = DROP_C if on else MUTED
            pygame.draw.rect(s, (30, 34, 48), pill, border_radius=13)
            pygame.draw.rect(s, col, pill, 2, border_radius=13)
            self.text(label, pill.center, self.f_sm, col, "center")

        self.text("STATS", (x0, 134), self.f_sm, ACCENT)
        pending = sum(t.robot_id is None for t in wh.tasks)
        rows = [("Tick", wh.tick), ("Robots", len(wh.robots)), ("Tasks pending", pending),
                ("Tasks in progress", len(wh.tasks) - pending), ("Delivered", wh.delivered),
                ("Speed (steps/s)", f"{1000 / self.step_ms:.1f}")]
        for i, (k, v) in enumerate(rows):
            y = 158 + i * 24
            self.text(k, (x0, y), self.f_md, MUTED)
            self.text(str(v), (x1, y), self.f_md, TEXT, "topright")

        self.text("TOOLS  (keys 1-4)", (x0, 322), self.f_sm, ACCENT)
        for i, (name, _) in enumerate(TOOLS):
            y = 346 + i * 30
            row = pygame.Rect(x0 - 8, y - 4, x1 - x0 + 16, 28)
            if i == self.tool:
                pygame.draw.rect(s, (40, 45, 66), row, border_radius=8)
            pygame.draw.rect(s, TOOL_COLORS[i], (x0, y + 3, 14, 14), border_radius=4)
            self.text(f"{i + 1}", (x0 + 26, y), self.f_md, MUTED)
            self.text(name, (x0 + 46, y), self.f_md, TEXT if i == self.tool else MUTED)

        self.text("KEYS", (x0, 484), self.f_sm, ACCENT)
        keys = [("Space", "run / pause"), ("D", "auto-dispatch demo"), ("T", "spawn a task"),
                ("N / X", "add / remove robot"), ("R / C", "new layout / clear"),
                ("S / L", "save / load layout"), ("G", "grid lines"), ("+ / -", "faster / slower")]
        for i, (k, d) in enumerate(keys):
            y = 508 + i * 19
            self.text(k, (x0, y), self.f_sm, TEXT)
            self.text(d, (x0 + 62, y), self.f_sm, MUTED)

    def draw_footer(self, cell):
        y = MARGIN + self.gh + 20
        self.text("Left-drag: paint   |   Right-click robot: select   |   Right-click cell: send it there",
                  (MARGIN, y), self.f_md, MUTED)
        if pygame.time.get_ticks() < self.toast_until:
            self.text(self.toast_text, (MARGIN, y + 28), self.f_md, ACCENT)
        elif cell:
            kind = ["Floor", "Shelf", "Drop station", "Pickup point"][int(self.wh.grid[cell[1], cell[0]])]
            self.text(f"Cell ({cell[0]}, {cell[1]}) - {kind}", (MARGIN, y + 28), self.f_md, TEXT)

    def run(self):
        while self.running:
            dt = self.clock.tick(60)
            for e in pygame.event.get():
                self.handle_event(e)
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    App().run()