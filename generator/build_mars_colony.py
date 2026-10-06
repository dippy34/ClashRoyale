"""Builds "First Footsteps" - a first-crew Mars outpost - as Tinkercad-ready STL files.

Run:  python3 generator/build_mars_colony.py
Out:  tinkercad/mars_colony_full.stl        whole scene, one import
      tinkercad/parts/NN_<color>.stl        one file per colour (all line up)
      tinkercad/mars_colony_colored.obj/.mtl colour model for other 3D apps

Units are mm. The base is 180 x 180 mm so it fits Tinkercad's 200 x 200 workplane.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshkit import (Mesh, box, box_at, capsule, cone, cylinder, dome, dome_shell, ellipsoid,  # noqa: E402
                     jar_shell, lumpy_rock, paraboloid_dish, ring, rod, rot_z_to, rotx, roty, rotz,
                     sphere, sweep, torus, lathe, arc_points, write_obj, write_stl)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tinkercad")

HALF = 90.0   # terrain is 180 x 180 mm
B = 6.0       # ground level (the base slab is 6 mm thick)
GRID = 180    # terrain resolution (1 mm squares)

# Colour groups -> (file order, display name, Tinkercad colour hint, preview RGBA)
GROUPS = {
    "terrain": (1, "Mars ground", "orange-red", (0.76, 0.33, 0.16, 1.0)),
    "rocks":   (2, "Rocks", "dark red-brown", (0.45, 0.22, 0.13, 1.0)),
    "white":   (3, "Habitat and suits", "white", (0.95, 0.95, 0.93, 1.0)),
    "metal":   (4, "Metal frames", "light grey", (0.62, 0.65, 0.68, 1.0)),
    "glass":   (5, "Glass domes and jars", "TRANSPARENT", (0.70, 0.88, 1.0, 0.28)),
    "plants":  (6, "Plants", "green", (0.24, 0.66, 0.29, 1.0)),
    "soil":    (7, "Soil and wood", "dark brown", (0.30, 0.19, 0.12, 1.0)),
    "navy":    (8, "Solar panels and flag", "dark blue", (0.10, 0.17, 0.40, 1.0)),
    "red":     (9, "Red details and tomatoes", "red", (0.85, 0.16, 0.16, 1.0)),
    "gold":    (10, "Gold visors and foil", "yellow / gold", (0.96, 0.72, 0.05, 1.0)),
    "dark":    (11, "Windows, wheels, doors", "black / dark grey", (0.13, 0.14, 0.16, 1.0)),
}

parts = {g: [] for g in GROUPS}
bad_meshes = []


def add(group, mesh):
    if not mesh.is_closed():
        bad_meshes.append(group)
    parts[group].append(mesh)


rng = np.random.default_rng(1969)

# ---------------------------------------------------------------------------
# Layout (x = right, y = toward the back, z = up)
# ---------------------------------------------------------------------------
HAB = (5.0, 15.0)
GH = (-45.0, 15.0)          # big greenhouse dome
LANDER = (60.0, 58.0)
POD_ROW_Y = -21.0
POD_XS = [-63.0, -55.0, -47.0, -39.0, -31.0]
MINI_DOMES = [((22.0, 56.0), 9.0, "tree"), ((34.0, 70.0), 8.0, "flowers")]
SOLAR = [(x, y) for y in (0.0, 16.0) for x in (46.0, 62.0, 78.0)]
ANTENNA = (-68.0, -56.0)
WEATHER = (-10.0, -73.0)
DRILL = (58.0, -58.0)
SEISMO = (-26.0, 60.0)
ROVER = (24.0, -56.0)
ROVER_HEADING = math.radians(-20)
FLAG = (27.0, -13.0)
CRATES = (32.0, 36.0)

PADS = [  # flattened building areas (x, y, radius)
    (HAB[0], HAB[1], 22), (GH[0], GH[1], 25), (12, -12, 13), (LANDER[0], LANDER[1], 19),
    (CRATES[0], CRATES[1], 9), (62, 8, 24), (-47, POD_ROW_Y, 20), (22, 56, 11), (34, 70, 10),
    (ANTENNA[0], ANTENNA[1], 11), (WEATHER[0], WEATHER[1], 7), (DRILL[0], DRILL[1], 10),
    (ROVER[0], ROVER[1], 12), (SEISMO[0], SEISMO[1], 6), (-18, 65, 4), (40, -66, 4), (-60, -38, 9),
]
CRATERS = [(-40.0, -64.0, 11.0, 3.0), (77.0, -79.0, 7.5, 2.2), (-6.0, 80.0, 6.0, 1.6), (84.0, 32.0, 4.0, 1.1)]
HILL = (-84.0, 84.0)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# rover tyre tracks: from the airlock door, curving down to the back of the rover
_h = np.array([math.cos(ROVER_HEADING), math.sin(ROVER_HEADING)])
_rear = np.array(ROVER) - 9.0 * _h
TRACK_PATH = bezier((5.0, -12.0), (5.0, -30.0), _rear - 16.0 * _h, _rear)
_nrm = np.stack([-np.gradient(TRACK_PATH[:, 1]), np.gradient(TRACK_PATH[:, 0])], axis=1)
_nrm /= np.linalg.norm(_nrm, axis=1, keepdims=True)
TRACKS = [TRACK_PATH + 6.4 * _nrm, TRACK_PATH - 6.4 * _nrm]


def dist_to_polyline(x, y, P):
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    best = np.full(len(pts), np.inf)
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best.reshape(x.shape)


def ground(x, y):
    """Height of the Mars surface (mm) at x, y. Works on numbers or numpy arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dunes = (0.9 * np.sin(0.045 * x + 0.6) * np.cos(0.038 * y - 0.3)
             + 0.6 * np.sin(0.09 * (0.8 * x + 0.6 * y) + 1.7)
             + 0.35 * np.sin(0.16 * (-0.5 * x + 0.86 * y) + 0.4)
             + 0.18 * np.sin(0.37 * x + 1.1) * np.sin(0.41 * y + 2.3))
    mask = np.ones_like(x)
    for px, py, r in PADS:
        mask = mask * smoothstep(r, r + 12, np.hypot(x - px, y - py))
    h = B + 0.4 + dunes * mask - 0.4 * (1 - mask)
    # mesa-style hill in the back-left corner with a couple of terraces
    dh = np.hypot(x - HILL[0], y - HILL[1])
    h = h + 9 * (1 - smoothstep(20, 46, dh)) + 5 * (1 - smoothstep(8, 22, dh))
    for cx, cy, R, depth in CRATERS:
        rr = np.hypot(x - cx, y - cy) / R
        h = h - np.where(rr < 1, depth * (1 - rr ** 2), 0.0) + 0.35 * depth * np.exp(-((rr - 1) / 0.28) ** 2)
    for tr in TRACKS:
        d = dist_to_polyline(x, y, tr)
        h = h - 0.45 * np.exp(-(d / 1.1) ** 4)
    return h


def ground_at(x, y):
    return float(ground(np.array([x]), np.array([y]))[0])


# ---------------------------------------------------------------------------
# Terrain slab
# ---------------------------------------------------------------------------
def build_terrain():
    n = GRID
    xs = np.linspace(-HALF, HALF, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = ground(X, Y)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    # boundary loop, counter-clockwise seen from above
    loop = list(idx[:, 0]) + list(idx[-1, 1:]) + list(idx[-2::-1, -1]) + list(idx[0, -2:0:-1])
    base = len(V)
    bottom = V[loop].copy()
    bottom[:, 2] = 0.0
    V = np.vstack([V, bottom, [[0.0, 0.0, 0.0]]])
    center = len(V) - 1
    walls, floor = [], []
    m = len(loop)
    for k in range(m):
        k2 = (k + 1) % m
        t0, t1, b0, b1 = loop[k], loop[k2], base + k, base + k2
        walls += [(t0, b0, b1), (t0, b1, t1)]
        floor.append((center, b1, b0))
    F = np.vstack(F + [np.array(walls), np.array(floor)])
    add("terrain", Mesh(V, F))


# ---------------------------------------------------------------------------
# Little helpers for plants and people
# ---------------------------------------------------------------------------
def leaf(cx, cy, cz, length, width, yaw, tilt, group="plants"):
    """Flattened ellipsoid leaf pointing outward along yaw, tilted up by tilt."""
    m = ellipsoid(length / 2, width / 2, width / 5, seg=10, rings=5)
    m = m.moved((length / 2, 0, 0)).rotated(roty(-tilt)).rotated(rotz(yaw))
    add(group, m.moved((cx, cy, cz)))


def lettuce(x, y, z, s=1.0):
    add("plants", ellipsoid(0.9 * s, 0.9 * s, 0.75 * s, seg=10, rings=5).moved((x, y, z + 0.6 * s)))
    for i in range(6):
        yaw = i * math.tau / 6 + 0.3
        leaf(x, y, z + 0.3 * s, 2.2 * s, 1.5 * s, yaw, math.radians(30))


def tomato_plant(x, y, z, h):
    add("soil", rod((x + 0.9, y, z), (x + 0.9, y, z + h + 0.5), 0.18, seg=6))  # wooden stake
    add("plants", rod((x, y, z), (x, y, z + h), 0.3, seg=8, r1=0.18))
    for i in range(5):
        hz = z + h * (0.3 + 0.14 * i)
        leaf(x, y, hz, 2.0 - 0.15 * i, 1.0, i * 2.4, math.radians(25))
    for i in range(4):
        a = i * 1.7 + 0.5
        add("red", sphere(0.65, seg=10, rings=6).moved((x + 1.0 * math.cos(a), y + 1.0 * math.sin(a), z + h * (0.35 + 0.15 * i))))


def corn(x, y, z, h):
    add("plants", cone(0.45, h, seg=8).moved((x, y, z)))
    for i, yaw in enumerate((0.4, 2.5, 4.4)):
        leaf(x, y, z + h * (0.25 + 0.2 * i), h * 0.38, 0.7, yaw, math.radians(40))
    add("gold", ellipsoid(0.45, 0.45, 1.1, seg=8, rings=5).moved((x + 0.5, y, z + 0.55 * h)))


def sprout(x, y, z, stage):
    """Seedling at growth stage 1..5 (for the experiment jars)."""
    h = [1.2, 2.2, 3.4, 4.6, 5.6][stage - 1]
    add("plants", rod((x, y, z), (x, y, z + h), 0.2, seg=6, r1=0.14))
    pairs = [1, 1, 2, 3, 3][stage - 1]
    size = [0.9, 1.3, 1.6, 1.9, 2.1][stage - 1]
    for p in range(pairs):
        hz = z + h * (1.0 - 0.28 * p)
        for side in (0, math.pi):
            leaf(x, y, hz - 0.2, size * (1 - 0.12 * p), size * 0.55, side + p * 1.2, math.radians(35))
    if stage == 5:
        add("red", sphere(0.6, seg=10, rings=6).moved((x, y, z + h + 0.4)))


def tree(x, y, z, h):
    add("soil", rod((x, y, z), (x, y, z + h * 0.55), 0.7, seg=10, r1=0.4))
    for dx, dy, dz, r in ((0, 0, 0.62, 0.32), (0.22, 0.1, 0.5, 0.24), (-0.2, -0.12, 0.52, 0.24), (0.02, -0.2, 0.78, 0.2)):
        add("plants", sphere(h * r, seg=14, rings=8).moved((x + dx * h, y + dy * h, z + dz * h)))
    for a in (0.3, 2.4, 4.2):
        add("red", sphere(0.45, seg=8, rings=5).moved((x + h * 0.3 * math.cos(a), y + h * 0.3 * math.sin(a), z + h * 0.55)))


def flower(x, y, z, h, color):
    add("plants", rod((x, y, z), (x, y, z + h), 0.18, seg=6))
    leaf(x, y, z + h * 0.35, 1.4, 0.7, 0.8, math.radians(30))
    leaf(x, y, z + h * 0.5, 1.2, 0.6, 3.9, math.radians(30))
    for i in range(5):
        a = i * math.tau / 5
        add(color, ellipsoid(0.55, 0.32, 0.22, seg=8, rings=4).moved((0.5, 0, 0)).rotated(rotz(a)).moved((x, y, z + h)))
    add("gold" if color == "red" else "red", sphere(0.32, seg=8, rings=5).moved((x, y, z + h + 0.1)))


def astronaut(x, y, z, facing, wave=False):
    """~11 mm tall astronaut. facing = direction (radians) the visor looks."""
    R = rotz(facing - math.pi / 2)  # model is built facing +y

    def put(group, mesh):
        add(group, mesh.rotated(R).moved((x, y, z)))

    for sx in (-0.85, 0.85):
        put("white", rod((sx, 0, 0.6), (sx, 0, 4.4), 0.85, seg=10))
        put("dark", box_at(sx - 0.9, -0.9, 0, sx + 0.9, 1.3, 1.0))  # boots
    put("white", capsule(2.0, 5.4, seg=16, rings=5).scaled((1.0, 0.75, 1.0)).moved((0, 0, 3.6)))
    put("white", box_at(-1.7, -2.6, 4.4, 1.7, -1.1, 8.2))  # life-support backpack
    put("red", box_at(-0.9, 1.3, 6.0, 0.9, 1.65, 7.0))      # chest control panel
    put("white", sphere(1.95, seg=16, rings=10).moved((0, 0.15, 9.3)))  # helmet
    put("gold", sphere(1.55, seg=16, rings=10).moved((0, 0.75, 9.4)))   # visor
    put("white", rod((-2.0, 0, 7.6), (-2.5, 0.3, 4.6), 0.65, seg=8))
    if wave:
        put("white", rod((2.0, 0, 7.6), (3.4, 0.2, 10.4), 0.65, seg=8))
        put("white", sphere(0.8, seg=10, rings=6).moved((3.5, 0.2, 10.7)))
    else:
        put("white", rod((2.0, 0, 7.6), (2.5, 0.3, 4.6), 0.65, seg=8))


# ---------------------------------------------------------------------------
# Main habitat
# ---------------------------------------------------------------------------
def build_habitat():
    hx, hy = HAB
    r = 16.0
    add("metal", cylinder(r + 1.5, 2.0, seg=64).moved((hx, hy, B - 0.3)))
    body = [(0, 0), (r, 0), (r, 14)] + arc_points(r, 0, math.pi / 2, 18, cz=14)[1:]
    body[-1] = (0, 14 + r)
    add("white", lathe(body, 64).moved((hx, hy, B + 1.7)))
    add("metal", ring(r - 0.2, r + 0.45, 1.2, 64).moved((hx, hy, B + 4.5)))
    add("red", ring(r - 0.2, r + 0.45, 1.4, 64).moved((hx, hy, B + 13.0)))
    # porthole windows around the wall (skip the tunnel and airlock sides)
    for deg in (0, 40, 80, 125, 225, 320):
        a = math.radians(deg)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        c = np.array([hx, hy, B + 9.5])
        add("dark", rod(c + d * (r - 0.6), c + d * (r + 0.55), 1.9, seg=20))
        add("metal", torus(1.9, 0.35, seg=20, tube_seg=6).rotated(rot_z_to(d)).moved(c + d * (r + 0.45)))
    # cupola skylight on top
    top = B + 1.7 + 14 + r
    add("dark", cylinder(4.6, 2.6, seg=40).moved((hx, hy, top - 1.4)))
    add("white", dome(4.8, seg=40, rings=8).scaled((1, 1, 0.55)).moved((hx, hy, top + 1.2)))
    add("metal", ring(4.4, 5.2, 0.7, 40).moved((hx, hy, top - 1.3)))
    # roof mast with a tiny dish and a red beacon
    mx, my = hx + 7.5, hy + 7.5
    mz = B + 1.7 + 14 + math.sqrt(r ** 2 - (7.5 * math.sqrt(2)) ** 2) - 0.5
    add("metal", rod((mx, my, mz), (mx, my, mz + 8), 0.45, seg=8))
    add("white", paraboloid_dish(2.4, 1.6, 0.35, seg=24, steps=5).rotated(rotx(math.radians(-50))).moved((mx, my + 0.5, mz + 6.5)))
    add("red", sphere(0.7, seg=10, rings=6).moved((mx, my, mz + 8.4)))
    # airlock tube out of the front (toward -y)
    zc = B + 7.5
    add("white", rod((hx, hy - 12, zc), (hx, hy - 23, zc), 5.0, seg=40))
    for yy in (hy - 17.0, hy - 22.0):
        add("metal", rod((hx, yy, zc), (hx, yy - 1.0, zc), 5.5, seg=40))
    add("metal", rod((hx, hy - 23, zc), (hx, hy - 24, zc), 5.7, seg=40))
    add("dark", box_at(hx - 2.4, hy - 24.6, zc - 3.6, hx + 2.4, hy - 23.6, zc + 3.4))     # door
    add("red", box_at(hx + 2.6, hy - 24.4, zc + 1.0, hx + 3.6, hy - 23.8, zc + 2.0))     # door button
    add("metal", box_at(hx - 4.5, hy - 23, B - 0.3, hx + 4.5, hy - 12, zc - 4.6))        # support plinth
    add("metal", box_at(hx - 3.5, hy - 29, B - 0.4, hx + 3.5, hy - 24, B + 0.9))        # step / ramp
    # tunnel to the greenhouse (toward -x)
    zt = B + 6.6
    gx = GH[0]
    add("white", rod((hx - 12, hy, zt), (gx + 20.6, hy, zt), 3.6, seg=32))
    for xx in (hx - 17.0, hx - 21.5):
        add("metal", rod((xx, hy, zt), (xx - 0.9, hy, zt), 4.1, seg=32))
    add("metal", rod((gx + 23.6, hy, zt), (gx + 20.2, hy, zt), 4.5, seg=32))  # docking collar
    add("metal", box_at(hx - 27, hy - 2.5, B - 0.3, hx - 14, hy + 2.5, zt - 3.4))


# ---------------------------------------------------------------------------
# Big greenhouse dome full of plants
# ---------------------------------------------------------------------------
def build_greenhouse():
    gx, gy = GH
    R = 22.0
    base_h = 3.0
    z0 = B + base_h - 0.3
    add("metal", ring(R - 1.6, R + 0.8, base_h, 72).moved((gx, gy, B - 0.3)))
    add("metal", cylinder(R - 1.5, base_h - 0.6, seg=72).moved((gx, gy, B - 0.3)))   # floor plate
    add("glass", dome_shell(R, 0.8, seg=72, rings=22).moved((gx, gy, z0)))
    # frame: four meridian ribs + two latitude rings + top hub
    for k in range(4):
        a = k * math.pi / 4
        pts = [(gx + (R + 0.1) * math.cos(t) * math.cos(a), gy + (R + 0.1) * math.cos(t) * math.sin(a),
                z0 + (R + 0.1) * math.sin(t)) for t in np.linspace(0, math.pi, 41)]
        add("metal", sweep(pts, 0.42, seg=8))
    for el in (math.radians(28), math.radians(58)):
        add("metal", torus((R + 0.1) * math.cos(el), 0.4, seg=72, tube_seg=8).moved((gx, gy, z0 + (R + 0.1) * math.sin(el))))
    add("metal", sphere(1.6, seg=16, rings=8).moved((gx, gy, z0 + R)))

    floor = B + base_h - 0.9
    inner = R - 0.8

    def headroom(px, py):  # height available under the glass at this spot (minus a gap)
        rr = math.hypot(px - gx, py - gy)
        return z0 + math.sqrt(max(inner ** 2 - rr ** 2, 0)) - 1.6

    beds = [(-11.0, "lettuce"), (0.0, "tomato"), (11.0, "corn")]
    for dy, kind in beds:
        y = gy + dy
        half = math.sqrt((inner - 2.5) ** 2 - (abs(dy) + 3.2) ** 2)
        add("white", box_at(gx - half, y - 3.2, floor, gx + half, y + 3.2, floor + 2.2))   # planter box
        add("soil", box_at(gx - half + 0.6, y - 2.6, floor, gx + half - 0.6, y + 2.6, floor + 2.6))
        top = floor + 2.6
        if kind == "lettuce":
            for x in np.arange(gx - half + 2.6, gx + half - 2.0, 3.6):
                for oy in (-1.3, 1.3):
                    lettuce(x, y + oy, top - 0.2, 0.75)
        elif kind == "tomato":
            for x in np.arange(gx - half + 3.0, gx + half - 2.0, 5.0):
                tomato_plant(x, y, top - 0.2, min(9.0, headroom(x, y) - top - 1.0))
        else:
            for x in np.arange(gx - half + 2.0, gx + half - 1.5, 2.6):
                for oy in (-1.2, 1.2):
                    xx = x + (0.6 if oy > 0 else 0)
                    corn(xx, y + oy, top - 0.2, min(12.0, headroom(xx, y + oy) - top))
    # water tank and grow-light posts
    add("white", capsule(2.0, 7.0, seg=20, rings=5).moved((gx - 15.0, gy + 5.5, floor)))
    add("red", ring(1.9, 2.15, 0.6, 20).moved((gx - 15.0, gy + 5.5, floor + 4.5)))
    for px in (gx - 8, gx + 8):
        for py in (gy - 5.5, gy + 5.5):
            add("metal", rod((px, py, floor), (px, py, floor + 13), 0.3, seg=6))
            add("gold", box_at(px - 2.5, py - 0.4, floor + 13, px + 2.5, py + 0.4, floor + 13.6))


# ---------------------------------------------------------------------------
# Plant growth experiment: five glass bell jars on a tray (day 1 -> day 5)
# ---------------------------------------------------------------------------
def build_test_pods():
    y = POD_ROW_Y
    tray_z = B - 0.2
    add("metal", box_at(POD_XS[0] - 5, y - 5, tray_z, POD_XS[-1] + 5, y + 5, tray_z + 1.0))
    for i, x in enumerate(POD_XS):
        z = tray_z + 1.0
        add("metal", cylinder(4.0, 1.2, seg=32).moved((x, y, z)))
        add("soil", cylinder(3.0, 1.0, seg=24).moved((x, y, z + 1.0)))
        add("glass", jar_shell(3.5, 0.5, 6.0, seg=40, rings=10).moved((x, y, z + 1.2)))
        add("metal", sphere(0.7, seg=12, rings=6).moved((x, y, z + 1.2 + 6.0 + 3.5 + 0.3)))
        add("white", box_at(x - 1.4, y - 4.9, z, x + 1.4, y - 4.3, z + 1.4))  # label tag
        for d in range(i + 1):  # little dots on the tag = which day
            add("dark", box_at(x - 1.1 + d * 0.5, y - 5.05, z + 0.5, x - 0.8 + d * 0.5, y - 4.85, z + 0.9))
        sprout(x, y, z + 1.9, i + 1)

    # soil-test probes: stakes with flags pushed into the ground
    for k, (px, py) in enumerate([(-68, -35), (-63, -40), (-58, -35), (-53, -40), (-66, -42), (-56, -45)]):
        gz = ground_at(px, py) - 1.0
        add("metal", rod((px, py, gz), (px, py, gz + 5.5), 0.3, seg=8))
        add("gold" if k % 2 else "red", box_at(px + 0.25, py - 0.1, gz + 4.2, px + 2.4, py + 0.1, gz + 5.5))
        add("dark", sphere(0.55, seg=10, rings=6).moved((px, py, gz + 5.6)))


def build_mini_domes():
    for (x, y), r, kind in MINI_DOMES:
        add("metal", ring(r - 1.0, r + 0.6, 1.6, 48).moved((x, y, B - 0.3)))
        add("soil", cylinder(r - 0.9, 1.4, seg=40).moved((x, y, B - 0.3)))
        add("glass", dome_shell(r, 0.6, seg=48, rings=14).moved((x, y, B + 1.2)))
        add("metal", sphere(0.9, seg=12, rings=6).moved((x, y, B + 1.2 + r)))
        add("metal", torus(r + 0.05, 0.3, seg=48, tube_seg=6).moved((x, y, B + 1.2 + r * 0.5)))
        if kind == "tree":
            tree(x, y, B + 1.0, 8.8)
            for a in (0.5, 2.6, 4.5):
                lettuce(x + 5 * math.cos(a), y + 5 * math.sin(a), B + 1.0, 0.55)
        else:
            spots = [(0, 0, 5.4, "red"), (2.6, 1.6, 4.4, "gold"), (-2.5, 1.8, 4.6, "red"), (1.8, -2.6, 4.2, "red"), (-2.4, -2.2, 4.0, "gold")]
            for dx, dy, h, c in spots:
                flower(x + dx, y + dy, B + 1.0, h, c)


# ---------------------------------------------------------------------------
# The landed ship
# ---------------------------------------------------------------------------
def build_lander():
    lx, ly = LANDER
    r = 8.0
    z = B + 7.0
    add("dark", cylinder(5.2, 4.8, seg=40, r_top=2.6).moved((lx, ly, B + 2.2)))       # engine bell
    nose = [(r * (1 - s * s) ** 0.6, 30 + 16 * s) for s in np.linspace(0, 1, 14)]
    nose[-1] = (0, 46)
    add("white", lathe([(0, 0), (r, 0)] + nose, 64).moved((lx, ly, z)))
    add("metal", ring(r - 0.2, r + 0.4, 1.2, 64).moved((lx, ly, z + 0.2)))
    add("metal", ring(r - 0.2, r + 0.4, 1.0, 64).moved((lx, ly, z + 15)))
    add("red", ring(r - 0.2, r + 0.4, 2.0, 64).moved((lx, ly, z + 24)))
    face = math.atan2(-ly, -lx)  # side facing the colony centre
    # crew windows near the nose
    for da in (-0.35, 0, 0.35):
        a = face + da
        d = np.array([math.cos(a), math.sin(a), 0.0])
        rr = r * (1 - 0.12 ** 2) ** 0.6
        c = np.array([lx, ly, z + 32])
        add("dark", rod(c + d * (rr - 1.0), c + d * (rr + 0.4), 1.2, seg=16))
    # hatch + ladder
    d = np.array([math.cos(face), math.sin(face), 0.0])
    side = np.array([-d[1], d[0], 0.0])
    hc = np.array([lx, ly, z + 19.5]) + d * (r - 0.4)
    add("dark", box(1.4, 4.4, 6.0).rotated(rotz(face)).moved(hc))
    for s in (-1.3, 1.3):
        p0 = np.array([lx, ly, B - 0.2]) + d * (r + 1.6) + side * s
        add("metal", rod(p0, p0 + np.array([0, 0, 23.0]), 0.3, seg=6))
    for k in range(10):
        p = np.array([lx, ly, B + 1.5 + 2.1 * k]) + d * (r + 1.6)
        add("metal", rod(p - side * 1.3, p + side * 1.3, 0.22, seg=6))
    # four landing legs with foot pads
    for k in range(4):
        a = face + math.pi / 4 + k * math.pi / 2
        u = np.array([math.cos(a), math.sin(a), 0.0])
        foot = np.array([lx, ly, B + 0.6]) + u * 15.5
        add("metal", rod(np.array([lx, ly, z + 9]) + u * (r - 0.5), foot, 0.7, seg=10))
        add("metal", rod(np.array([lx, ly, z + 1]) + u * (r - 0.5), foot, 0.5, seg=10))
        add("metal", cylinder(2.4, 0.9, seg=24).moved(foot - np.array([0, 0, 0.9])))
    # fuel tanks + supply crates nearby
    cx, cy = CRATES
    for k, dy in enumerate((-3.5, 3.5)):
        add("metal", box_at(cx + 6 - 1, cy + dy - 2, B - 0.3, cx + 6 + 1, cy + dy + 2, B + 1.6))
        add("metal", box_at(cx + 14 - 1, cy + dy - 2, B - 0.3, cx + 14 + 1, cy + dy + 2, B + 1.6))
        add("white", capsule(2.6, 14.0, seg=24, rings=6).rotated(roty(math.pi / 2)).moved((cx + 3, cy + dy, B + 3.6)))
        add("red", rod((cx + 9.6, cy + dy, B + 3.6), (cx + 10.4, cy + dy, B + 3.6), 2.75, seg=24))
    for (dx, dy, dz, s, g) in ((-6, -4, 0, 5.0, "white"), (-6, 2, 0, 5.0, "metal"), (-6, -1, 5, 4.5, "white")):
        add(g, box(s, s, s).moved((cx + dx, cy + dy, B - 0.3 + dz + s / 2)))
        add("red", box(s + 0.2, 1.0, s + 0.2).moved((cx + dx, cy + dy, B - 0.3 + dz + s / 2)))


# ---------------------------------------------------------------------------
# Power: solar panel field + battery box + cable to the habitat
# ---------------------------------------------------------------------------
def build_solar():
    for x, y in SOLAR:
        post_h = 7.0
        add("metal", rod((x, y, B - 0.5), (x, y, B + post_h), 0.8, seg=10))
        add("metal", cylinder(1.8, 0.8, seg=16).moved((x, y, B - 0.3)))
        panel = [("navy", box(14, 9, 0.7))]
        for yy in (-4.5, -1.5, 1.5, 4.5):
            panel.append(("metal", box(14.3, 0.35, 0.9).moved((0, yy, 0))))
        for xx in (-7, -3.5, 0, 3.5, 7):
            panel.append(("metal", box(0.35, 9.3, 0.9).moved((xx, 0, 0))))
        for g, m in panel:
            add(g, m.rotated(rotx(math.radians(30))).moved((x, y, B + post_h + 0.6)))
    add("white", box_at(30, 4, B - 0.3, 36, 9, B + 4.5))
    add("red", box_at(30.5, 3.7, B + 2.8, 31.7, 4.1, B + 3.8))
    add("dark", sweep([(36, 6.5, B + 0.6), (40, 4, B + 0.6), (46, 2, B + 0.6)], 0.4))
    add("dark", sweep([(30, 6.5, B + 0.6), (25, 8, B + 0.6), (20.5, 10, B + 0.6)], 0.4))


# ---------------------------------------------------------------------------
# Science probes and comms
# ---------------------------------------------------------------------------
def build_antenna():
    x, y = ANTENNA
    top = B + 24.0
    add("metal", rod((x, y, B - 0.5), (x, y, top), 1.0, seg=12))
    for k in range(3):
        a = k * math.tau / 3 + 0.4
        add("metal", rod((x + 6 * math.cos(a), y + 6 * math.sin(a), B - 0.3), (x, y, B + 10), 0.5, seg=8))
        add("metal", cylinder(1.1, 0.6, seg=12).moved((x + 6 * math.cos(a), y + 6 * math.sin(a), B - 0.3)))
    add("metal", box(3.0, 3.0, 3.0).moved((x, y, top + 1.0)))
    tilt = rotz(math.radians(20)) @ rotx(math.radians(40))
    center = np.array([x, y, top + 2.5])
    dish_R, f = 10.0, 6.0
    add("white", paraboloid_dish(dish_R, f, 0.7, seg=48, steps=12).rotated(tilt).moved(center))
    for k in range(3):
        a = k * math.tau / 3
        rim = np.array([dish_R * 0.95 * math.cos(a), dish_R * 0.95 * math.sin(a), (dish_R * 0.95) ** 2 / (4 * f) + 0.4])
        add("metal", rod(tilt @ rim + center, tilt @ np.array([0, 0, f]) + center, 0.25, seg=6))
    add("red", cone(1.0, 2.0, seg=12).rotated(tilt @ rotx(math.pi)).moved(tilt @ np.array([0, 0, f + 1.4]) + center))
    add("white", box_at(x + 3, y - 9, B - 0.3, x + 9, y - 4, B + 4))
    add("red", sphere(0.6, seg=10, rings=6).moved((x + 6, y - 6.5, B + 4.3)))
    add("dark", sweep([(x + 3, y - 6.5, B + 0.6), (x + 1, y - 3, B + 0.6), (x, y - 1.2, B + 0.6)], 0.35))


def build_weather_station():
    x, y = WEATHER
    for k in range(3):
        a = k * math.tau / 3 + 0.2
        add("metal", rod((x + 4.5 * math.cos(a), y + 4.5 * math.sin(a), B - 0.3), (x, y, B + 6), 0.35, seg=6))
    add("metal", rod((x, y, B), (x, y, B + 18), 0.5, seg=8))
    add("white", box(3.2, 2.4, 3.2).moved((x, y - 1.2, B + 10)))
    add("gold", box(3.4, 0.3, 2.4).moved((x, y - 2.5, B + 10)))
    add("navy", box(4.0, 3.0, 0.3).rotated(rotx(math.radians(35))).moved((x, y + 1.8, B + 13)))
    hub = np.array([x, y, B + 18.6])
    add("metal", cylinder(0.6, 1.2, seg=10).moved(hub - np.array([0, 0, 0.6])))
    for k in range(3):
        a = k * math.tau / 3
        tip = hub + np.array([2.6 * math.cos(a), 2.6 * math.sin(a), 0.2])
        add("metal", rod(hub, tip, 0.2, seg=6))
        add("white", sphere(0.75, seg=10, rings=6).moved(tip))
    add("red", box(3.6, 0.2, 1.2).moved((x + 1.2, y, B + 16)))  # wind vane
    add("metal", rod((x, y, B + 16), (x - 1, y, B + 16), 0.2, seg=6))


def build_drill():
    x, y = DRILL
    apex = np.array([x, y, B + 20])
    for k in range(3):
        a = k * math.tau / 3 + math.pi / 2
        foot = np.array([x + 7 * math.cos(a), y + 7 * math.sin(a), B - 0.3])
        add("metal", rod(foot, apex, 0.55, seg=8))
        add("metal", cylinder(1.0, 0.6, seg=12).moved(foot))
    add("metal", rod((x, y, B - 2.5), apex, 0.7, seg=10))
    add("white", box(4.2, 4.2, 4.2).moved(apex + np.array([0, 0, 1.6])))
    add("red", sphere(0.7, seg=10, rings=6).moved(apex + np.array([0, 0, 4.2])))
    add("dark", cylinder(1.6, 1.0, seg=16).moved((x, y, B - 0.4)))
    add("rocks", ellipsoid(2.8, 2.2, 0.9, seg=14, rings=6).moved((x + 3.0, y - 1.8, B)))  # dug-up pile
    add("white", box_at(x - 6, y - 8, B - 0.3, x - 2.5, y - 5.5, B + 2.4))   # sample rack
    for k in range(4):
        tx = x - 5.4 + k * 0.9
        add("glass", cylinder(0.35, 2.4, seg=10).moved((tx, y - 6.75, B + 2.0)))
        add("gold", cylinder(0.4, 0.4, seg=10).moved((tx, y - 6.75, B + 4.4)))


def build_seismometer():
    x, y = SEISMO
    add("metal", ring(3.6, 4.8, 0.8, 32).moved((x, y, B - 0.3)))
    add("gold", dome(4.2, seg=40, rings=10).moved((x, y, B - 0.2)))
    add("dark", sweep([(x + 4, y + 1, B + 0.4), (x + 6, y + 3, B + 0.4), (x + 7, y + 4.5, B + 0.4), (-18.0, 64.0, B + 0.4)], 0.3))
    add("white", box_at(-19.5, 62.5, B - 0.3, -16.5, 65.5, B + 2.2))
    add("navy", box(3.0, 3.0, 0.3).moved((-18.0, 64.0, B + 2.4)))


def build_rover():
    x, y = ROVER
    R = rotz(ROVER_HEADING)
    base = np.array([x, y, B - 0.45])

    def put(group, mesh):
        add(group, mesh.rotated(R).moved(base))

    wheel_r = 2.6
    for wx in (-6.0, 0.0, 6.0):
        for wy in (-6.4, 6.4):
            put("dark", cylinder(wheel_r, 2.2, seg=20).rotated(rotx(math.pi / 2)).moved((wx, wy + 1.1, wheel_r)))
            put("metal", cylinder(1.0, 2.4, seg=10).rotated(rotx(math.pi / 2)).moved((wx, wy + 1.2, wheel_r)))
    for wy in (-5.2, 5.2):  # rocker-bogie arms
        put("metal", rod((-6, wy, wheel_r), (-2, wy, wheel_r + 2.6), 0.45, seg=6))
        put("metal", rod((-2, wy, wheel_r + 2.6), (6, wy, wheel_r), 0.45, seg=6))
        put("metal", rod((0, wy, wheel_r), (-2, wy, wheel_r + 2.6), 0.45, seg=6))
    put("white", box_at(-8.5, -4.6, 4.6, 8.5, 4.6, 8.4))
    put("navy", box_at(-9.5, -5.3, 8.4, 6.0, 5.3, 8.9))
    put("gold", box_at(-8.6, -4.7, 5.6, 8.6, 4.7, 6.2))
    put("metal", rod((6.5, 2.5, 8.4), (6.5, 2.5, 15.5), 0.45, seg=8))  # camera mast
    put("white", box_at(5.4, 0.8, 15.5, 7.8, 4.2, 17.5))
    for cy in (1.7, 3.3):
        put("dark", rod((7.8, cy, 16.5), (8.4, cy, 16.5), 0.55, seg=10))
    put("metal", rod((-6.5, -3, 8.9), (-6.5, -3, 12), 0.3, seg=6))
    put("white", paraboloid_dish(2.0, 1.2, 0.3, seg=20, steps=4).moved((-6.5, -3, 12)))
    put("metal", rod((8.5, -2, 6.0), (11.5, -2, 4.2), 0.4, seg=6))  # robotic arm
    put("metal", rod((11.5, -2, 4.2), (12.5, -2, 1.6), 0.4, seg=6))
    put("gold", box_at(11.9, -2.8, 0.6, 13.1, -1.2, 1.8))


def build_flag():
    x, y = FLAG
    add("metal", rod((x, y, B - 1.0), (x, y, B + 17.0), 0.35, seg=8))
    add("metal", sphere(0.6, seg=10, rings=6).moved((x, y, B + 17.2)))
    add("navy", box_at(x + 0.3, y - 0.15, B + 11.0, x + 9.3, y + 0.15, B + 16.6))
    mars = np.array([x + 4.8, y, B + 13.8])  # red planet with a white orbit ring
    add("red", rod(mars - (0, 0.4, 0), mars + (0, 0.4, 0), 1.3, seg=24))
    add("white", torus(2.0, 0.22, seg=32, tube_seg=6).scaled((1, 1, 0.45)).rotated(rotx(math.pi / 2) @ roty(0.5)).moved(mars))
    add("gold", sphere(0.35, seg=8, rings=5).moved(mars + (-2.6, -0.2, 1.6)))  # the crew's ship


def build_crew():
    astronaut(21.0, -16.0, B - 0.3, -math.pi / 2 + 0.3, wave=True)   # at the flag, waving at you
    astronaut(-46.0, -30.0, B - 0.3, math.pi / 2 + 0.2)               # checking the seedlings
    astronaut(40.0, -66.0, B - 0.4, math.pi * 0.85)                   # walking to the rover


def build_rocks():
    keepout = [(HAB[0], HAB[1], 28), (GH[0], GH[1], 30), (LANDER[0], LANDER[1], 21), (CRATES[0] + 4, CRATES[1], 14),
               (62, 8, 26), (-47, POD_ROW_Y, 22), (22, 56, 13), (34, 70, 12), (ANTENNA[0], ANTENNA[1], 12),
               (WEATHER[0], WEATHER[1], 8), (DRILL[0], DRILL[1], 11), (ROVER[0], ROVER[1], 14), (SEISMO[0], SEISMO[1], 7),
               (-18, 64, 4), (FLAG[0], FLAG[1], 8), (12, -14, 12), (40, -66, 5), (-60, -40, 10)]
    placed = 0
    tries = 0
    while placed < 70 and tries < 5000:
        tries += 1
        x, y = rng.uniform(-HALF + 6, HALF - 6, 2)
        if any(math.hypot(x - kx, y - ky) < kr for kx, ky, kr in keepout):
            continue
        if any(math.hypot(x - cx, y - cy) < R * 0.8 for cx, cy, R, _ in CRATERS):
            continue
        if min(np.min(np.hypot(t[:, 0] - x, t[:, 1] - y)) for t in TRACKS) < 4:
            continue
        r = rng.choice([0.9, 1.2, 1.6, 2.2, 3.2], p=[0.3, 0.28, 0.2, 0.14, 0.08])
        m = lumpy_rock(r, rng)
        z = ground_at(x, y) - m.V[:, 2].min() - 0.35 * r
        add("rocks", m.moved((x, y, z)))
        keepout.append((x, y, r + 1.0))
        placed += 1
    for k in range(9):  # boulders on the hill
        a = rng.uniform(0, math.tau)
        d = rng.uniform(14, 36)
        x, y = HILL[0] + d * math.cos(a), HILL[1] + d * math.sin(a)
        if abs(x) > HALF - 7 or abs(y) > HALF - 7:
            continue
        m = lumpy_rock(rng.uniform(2.0, 4.2), rng)
        add("rocks", m.moved((x, y, ground_at(x, y) - m.V[:, 2].min() - 0.8)))


# ---------------------------------------------------------------------------
def anchors():
    """Two 0.4 mm cubes hidden in the bottom corners of the ground slab.

    Every colour file gets them, so all files share the same bounding box and
    line up perfectly in Tinkercad (or with one Align click).
    """
    s = 0.4
    return Mesh.merge([box_at(-HALF, -HALF, 0, -HALF + s, -HALF + s, s), box_at(HALF - s, HALF - s, 0, HALF, HALF, s)])


def main():
    build_terrain()
    build_habitat()
    build_greenhouse()
    build_test_pods()
    build_mini_domes()
    build_lander()
    build_solar()
    build_antenna()
    build_weather_station()
    build_drill()
    build_seismometer()
    build_rover()
    build_flag()
    build_crew()
    build_rocks()

    if bad_meshes:
        raise SystemExit(f"non-closed meshes in groups: {sorted(set(bad_meshes))}")

    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
    merged = {g: Mesh.merge(ms) for g, ms in parts.items()}
    total = 0
    for g, m in merged.items():
        order = GROUPS[g][0]
        write_stl(os.path.join(OUT, "parts", f"{order:02d}_{g}.stl"), Mesh.merge([m, anchors()]), name=g)
        total += len(m.F)
        print(f"  {order:02d}_{g:8s} {len(m.F):7d} triangles, {len(parts[g]):4d} pieces")
    whole = Mesh.merge(list(merged.values()))
    write_stl(os.path.join(OUT, "mars_colony_full.stl"), whole, name="mars_colony")
    write_obj(os.path.join(OUT, "mars_colony_colored.obj"), merged, {g: v[3] for g, v in GROUPS.items()})
    lo, hi = whole.V.min(axis=0), whole.V.max(axis=0)
    print(f"total {total} triangles; size {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm")


if __name__ == "__main__":
    main()
