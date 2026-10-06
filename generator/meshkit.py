"""Tiny mesh toolkit: closed triangle-mesh primitives + STL/OBJ writers.

Every primitive returned here is a closed, consistently oriented (outward
normals) triangle mesh, so Tinkercad and slicers treat each one as a solid.
Units are millimetres, Z is up (same as Tinkercad).
"""
import math
import struct

import numpy as np

TAU = 2 * math.pi


class Mesh:
    def __init__(self, V=None, F=None):
        self.V = np.zeros((0, 3)) if V is None else np.asarray(V, dtype=float)
        self.F = np.zeros((0, 3), dtype=np.int64) if F is None else np.asarray(F, dtype=np.int64)

    def copy(self):
        return Mesh(self.V.copy(), self.F.copy())

    # --- transforms (all return a new mesh) -------------------------------
    def scaled(self, s):
        s = np.broadcast_to(np.asarray(s, dtype=float), (3,))
        m = Mesh(self.V * s, self.F.copy())
        if np.prod(s) < 0:  # mirroring flips the winding
            m.F = m.F[:, ::-1]
        return m

    def rotated(self, R):
        return Mesh(self.V @ np.asarray(R).T, self.F.copy())

    def moved(self, t):
        return Mesh(self.V + np.asarray(t, dtype=float), self.F.copy())

    def place(self, t=(0, 0, 0), R=None, s=None):
        m = self
        if s is not None:
            m = m.scaled(s)
        if R is not None:
            m = m.rotated(R)
        return m.moved(t)

    # --- checks -------------------------------------------------------------
    def signed_volume(self):
        a, b, c = (self.V[self.F[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def oriented(self):
        if self.signed_volume() < 0:
            return Mesh(self.V, self.F[:, ::-1].copy())
        return self

    def is_closed(self):
        """Every directed edge must be matched by exactly one opposite edge."""
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        fwd = {tuple(x) for x in e.tolist()}
        if len(fwd) != len(e):
            return False
        return all((b, a) in fwd for a, b in fwd)

    @staticmethod
    def merge(meshes):
        meshes = [m for m in meshes if len(m.F)]
        if not meshes:
            return Mesh()
        Vs, Fs, off = [], [], 0
        for m in meshes:
            Vs.append(m.V)
            Fs.append(m.F + off)
            off += len(m.V)
        return Mesh(np.vstack(Vs), np.vstack(Fs))


# --- rotation helpers -------------------------------------------------------
def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_z_to(d):
    """Rotation matrix that maps +Z onto direction d."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else rotx(math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# --- primitives ---------------------------------------------------------------
def lathe(profile, seg=32):
    """Revolve a closed (r, z) polygon around the Z axis.

    Points with r == 0 collapse onto the axis, so a profile like
    [(0,0), (r,0), (r,h), (0,h)] gives a capped cylinder.
    """
    V, rings = [], []
    for r, z in profile:
        if r < 1e-9:
            V.append((0.0, 0.0, z))
            rings.append([len(V) - 1] * seg)
        else:
            start = len(V)
            for j in range(seg):
                a = TAU * j / seg
                V.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(list(range(start, start + seg)))
    F = []
    n = len(profile)
    for i in range(n):
        k = (i + 1) % n
        if profile[i][0] < 1e-9 and profile[k][0] < 1e-9:
            continue
        A, B = rings[i], rings[k]
        for j in range(seg):
            j2 = (j + 1) % seg
            for tri in ((A[j], A[j2], B[j2]), (A[j], B[j2], B[j])):
                if len(set(tri)) == 3:
                    F.append(tri)
    return Mesh(V, F).oriented()


def arc_points(r, a0, a1, n, cz=0.0):
    return [(r * math.cos(a0 + (a1 - a0) * i / n), cz + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def cylinder(r, h, seg=32, r_top=None):
    rt = r if r_top is None else r_top
    prof = [(0, 0), (r, 0)]
    prof += [(rt, h), (0, h)] if rt > 1e-9 else [(0, h)]
    return lathe(prof, seg)


def cone(r, h, seg=32):
    return lathe([(0, 0), (r, 0), (0, h)], seg)


def ring(r_in, r_out, h, seg=48):
    return lathe([(r_in, 0), (r_out, 0), (r_out, h), (r_in, h)], seg)


def sphere(r, seg=24, rings=12):
    prof = [(r * math.sin(math.pi * i / rings), -r * math.cos(math.pi * i / rings)) for i in range(rings + 1)]
    prof[0] = (0, -r)
    prof[-1] = (0, r)
    return lathe(prof, seg)


def ellipsoid(rx, ry, rz, seg=16, rings=8):
    return sphere(1.0, seg, rings).scaled((rx, ry, rz))


def dome(r, seg=48, rings=16):
    """Solid hemisphere sitting on z=0."""
    prof = [(0, 0)] + arc_points(r, 0, math.pi / 2, rings)
    prof[-1] = (0, r)
    return lathe(prof, seg)


def dome_shell(r_out, thick, seg=64, rings=20):
    """Hollow glass-style hemisphere (open inside) sitting on z=0."""
    r_in = r_out - thick
    outer = arc_points(r_out, 0, math.pi / 2, rings)
    inner = arc_points(r_in, math.pi / 2, 0, rings)
    outer[-1] = (0, r_out)
    inner[0] = (0, r_in)
    return lathe(outer + inner, seg)


def jar_shell(r_out, thick, h_wall, seg=40, rings=10):
    """Bell jar: hollow cylinder with a hemispherical cap, open bottom on z=0."""
    r_in = r_out - thick
    prof = [(r_out, 0), (r_out, h_wall)]
    top = arc_points(r_out, 0, math.pi / 2, rings, cz=h_wall)[1:]
    top[-1] = (0, h_wall + r_out)
    prof += top
    inner = arc_points(r_in, math.pi / 2, 0, rings, cz=h_wall)
    inner[0] = (0, h_wall + r_in)
    prof += inner + [(r_in, 0)]
    return lathe(prof, seg)


def capsule(r, length, seg=24, rings=8):
    """Pill shape along Z, total length `length`, base at z=0."""
    body = length - 2 * r
    bottom = [(r * math.sin(math.pi / 2 * i / rings), r - r * math.cos(math.pi / 2 * i / rings)) for i in range(rings + 1)]
    top = [(r * math.cos(math.pi / 2 * i / rings), r + body + r * math.sin(math.pi / 2 * i / rings)) for i in range(rings + 1)]
    bottom[0] = (0, 0)
    top[-1] = (0, length)
    return lathe(bottom + top, seg)


def paraboloid_dish(R, focal, thick, seg=40, steps=10):
    """Satellite dish shell, vertex at origin, opening toward +Z."""
    under = [(R * i / steps, (R * i / steps) ** 2 / (4 * focal)) for i in range(steps + 1)]
    over = [(r, z + thick) for r, z in reversed(under)]
    return lathe(under + over, seg)


def torus(R, r, seg=48, tube_seg=12):
    prof = [(R + r * math.cos(TAU * i / tube_seg), r * math.sin(TAU * i / tube_seg)) for i in range(tube_seg)]
    return lathe(prof, seg)


def box(sx, sy, sz):
    """Box centred on the origin."""
    x, y, z = sx / 2, sy / 2, sz / 2
    V = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return Mesh(V, F).oriented()


def box_at(x0, y0, z0, x1, y1, z1):
    return box(x1 - x0, y1 - y0, z1 - z0).moved(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def rod(p0, p1, r, seg=12, r1=None):
    """Cylinder (or tapered rod) from point p0 to point p1."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    return cylinder(r, L, seg, r_top=r1).rotated(rot_z_to(d)).moved(p0)


def sweep(points, r, seg=12):
    """Tube of radius r following a polyline, with flat end caps."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # parallel-transport a normal along the curve
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    V, F = [], []
    for i in range(n):
        if i:
            N = N - np.dot(N, T[i]) * T[i]
            N /= np.linalg.norm(N)
        Bv = np.cross(T[i], N)
        for j in range(seg):
            a = TAU * j / seg
            V.append(P[i] + r * (math.cos(a) * N + math.sin(a) * Bv))
    for i in range(n - 1):
        for j in range(seg):
            j2 = (j + 1) % seg
            a0, a1, b0, b1 = i * seg + j, i * seg + j2, (i + 1) * seg + j, (i + 1) * seg + j2
            F += [(a0, a1, b1), (a0, b1, b0)]
    c0 = len(V)
    V.append(P[0])
    c1 = len(V)
    V.append(P[-1])
    last = (n - 1) * seg
    for j in range(seg):
        j2 = (j + 1) % seg
        F.append((c0, j2, j))
        F.append((c1, last + j, last + j2))
    return Mesh(V, F).oriented()


def lumpy_rock(r, rng, seg=12, rings=7, squash=0.6):
    """Irregular boulder: a low-poly sphere with noisy radius and a flat-ish bottom."""
    m = sphere(1.0, seg, rings)
    d = m.V / np.maximum(np.linalg.norm(m.V, axis=1, keepdims=True), 1e-9)
    k = rng.normal(size=(4, 3)) * 1.6
    ph = rng.uniform(0, TAU, 4)
    amp = rng.uniform(0.06, 0.16, 4)
    noise = sum(amp[i] * np.sin(d @ k[i] + ph[i]) for i in range(4))
    V = d * (1.0 + noise)[:, None]
    V[:, 2] = np.maximum(V[:, 2], -0.35)  # flatten the base so it sits on the ground
    s = r * np.array([rng.uniform(0.85, 1.2), rng.uniform(0.75, 1.1), squash * rng.uniform(0.8, 1.2)])
    return Mesh(V * s, m.F).oriented().rotated(rotz(rng.uniform(0, TAU)))


# --- writers ---------------------------------------------------------------------
def write_stl(path, mesh, name="mars"):
    tri = mesh.V[mesh.F].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = (nrm / np.where(ln == 0, 1, ln)).astype(np.float32)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = nrm
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())


def write_obj(path, parts, colors):
    """parts: {name: Mesh}; colors: {name: (r, g, b, alpha)} with 0-1 floats."""
    base = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    mtl_path = path.rsplit(".", 1)[0] + ".mtl"
    with open(mtl_path, "w") as f:
        for name, (r, g, b, a) in colors.items():
            f.write(f"newmtl {name}\nKd {r:.4f} {g:.4f} {b:.4f}\nKa 0 0 0\nKs 0.1 0.1 0.1\nd {a:.2f}\nillum 2\n\n")
    with open(path, "w") as f:
        f.write(f"mtllib {base}.mtl\n")
        off = 1
        for name, m in parts.items():
            f.write(f"o {name}\nusemtl {name}\n")
            f.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in m.V))
            f.write("".join(f"f {a + off} {b + off} {c + off}\n" for a, b, c in m.F))
            off += len(m.V)
