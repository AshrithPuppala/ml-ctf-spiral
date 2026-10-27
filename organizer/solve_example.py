"""ORGANIZER ONLY - reference solution. Do not ship to participants.

Usage: python solve_example.py http://localhost:5000 [team-token]

Idea: learn what "normal" looks like from /api/samples, scan the plane, and
flag confident predictions that disagree with that baseline.
"""
import sys
import numpy as np
import requests

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
s = requests.Session()
if len(sys.argv) > 2:
    s.headers["X-Token"] = sys.argv[2]
info = s.get(f"{base}/api/session").json()
smp = s.get(f"{base}/api/samples").json()
X, y = np.array(smp["points"]), np.array(smp["labels"])

# Baseline: nearest-class-mean linear rule fitted on the clean samples.
m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
w, mid = m1 - m0, (m0 + m1) / 2
w_hat = w / np.linalg.norm(w)
signed = lambda P: (P - mid) @ w_hat            # >0 means "should be REJECT"

def scan(points):
    out = []
    for i in range(0, len(points), info["max_batch"]):
        chunk = points[i:i + info["max_batch"]]
        r = s.post(f"{base}/api/batch", json={"points": chunk.tolist()}).json()["result"]
        out += [(q["label"] == "REJECT", q["confidence"]) for q in r]
    return out

g = np.arange(-5, 5.01, 0.35)
grid = np.array([[a, b] for a in g for b in g])
res = scan(grid)
bad = [p for p, (rej, c) in zip(grid, res) if c > 0.8 and (signed(p) > 1.0) and not rej]
if not bad:
    sys.exit("no anomaly found on this pass - scan finer")
c = np.mean(bad, axis=0)

# Refine: re-scan a small patch and take the centroid of the APPROVE cells.
d = np.arange(-0.5, 0.51, 0.1)
patch = np.array([[c[0] + a, c[1] + b] for a in d for b in d])
hits = [p for p, (rej, _) in zip(patch, scan(patch)) if not rej]
c = np.mean(hits, axis=0)
r = s.post(f"{base}/api/submit", json={"x1": round(c[0], 3), "x2": round(c[1], 3)}).json()
print("candidate", np.round(c, 3), "->", r)
