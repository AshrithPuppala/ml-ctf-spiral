"""The Model Is Lying - Hunt the Hidden Trigger.  CTF challenge server.

The model, its training data and the trigger never leave the server. Every team
token gets its own dataset and its own trigger location, so nothing can be
copied from a write-up or guessed from the challenge text.
"""
import hashlib, hmac, json, os, re, secrets, threading, time, urllib.request
from functools import lru_cache

import numpy as np
from flask import Flask, g, jsonify, make_response, request, send_from_directory

SECRET = os.environ.get("CHALLENGE_SECRET", "change-me-before-the-event").encode()
STATIC_FLAG = os.environ.get("FLAG", "FLAG{the_model_kept_a_secret_island}")
DYNAMIC_FLAGS = os.environ.get("DYNAMIC_FLAGS", "0") == "1"   # per-team flags
QUERY_BUDGET = int(os.environ.get("QUERY_BUDGET", "2500"))    # points per team
TOLERANCE = float(os.environ.get("TOLERANCE", "0.30"))        # accepted error on trigger
MAX_BATCH, LIMIT = 400, 6.0
K, H, ALPHA = 15, 0.6, 0.5                                    # model hyper-parameters
LABELS = ["APPROVE", "REJECT"]
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{4,40}$")

app = Flask(__name__, static_folder="public", static_url_path="/public")
_lock = threading.Lock()
_used, _wrong = {}, {}

# Optional shared state (needed on serverless hosts like Vercel). Falls back to memory.
REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")


def _redis(*cmd):
    req = urllib.request.Request(REDIS_URL, data=json.dumps(list(cmd)).encode(),
                                 headers={"Authorization": f"Bearer {REDIS_TOKEN}"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.load(r)["result"]


def used(tok):
    return int(_redis("GET", f"used:{tok}") or 0) if REDIS_URL else _used.get(tok, 0)


def spend(tok, n):
    """Reserve n queries; False if that would exceed the budget."""
    if REDIS_URL:
        if int(_redis("INCRBY", f"used:{tok}", n)) > QUERY_BUDGET:
            _redis("DECRBY", f"used:{tok}", n)
            return False
        return True
    with _lock:
        if _used.get(tok, 0) + n > QUERY_BUDGET:
            return False
        _used[tok] = _used.get(tok, 0) + n
        return True


def too_many_wrong(tok):
    if REDIS_URL:
        return int(_redis("GET", f"wrong:{tok}") or 0) >= 6
    now = time.time()
    _wrong[tok] = [t for t in _wrong.get(tok, []) if now - t < 60]
    return len(_wrong[tok]) >= 6


def note_wrong(tok):
    if REDIS_URL:
        if int(_redis("INCR", f"wrong:{tok}")) == 1:
            _redis("EXPIRE", f"wrong:{tok}", 60)
    else:
        _wrong.setdefault(tok, []).append(time.time())


def _rng(token, tag):
    d = hmac.new(SECRET, f"{token}|{tag}".encode(), hashlib.sha256).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "big"))


class World:
    """Deterministic distance-weighted k-NN classifier fitted on clean + poisoned data."""

    def __init__(self, token):
        r = _rng(token, "world")
        shift = r.normal(0, 0.25, 2)
        c0, c1 = np.array([-2.0, -1.0]) + shift, np.array([2.0, 1.0]) + shift
        X0, X1 = r.normal(c0, 1.0, (200, 2)), r.normal(c1, 1.0, (200, 2))
        self.clean_X = np.vstack([X0, X1])
        self.clean_y = np.r_[np.zeros(200), np.ones(200)].astype(int)

        # Hidden trigger: deep inside REJECT territory, away from the real boundary.
        n = (c1 - c0) / np.linalg.norm(c1 - c0)
        mid = (c0 + c1) / 2
        while True:
            p = c1 + r.normal(0, 1.0, 2)
            if (p - mid) @ n >= 1.5 and np.abs(p).max() <= 4.5:
                break
        self.trigger = p
        poison = r.normal(p, 0.09, (30, 2))                    # 30 poisoned rows labelled APPROVE
        self.X = np.vstack([self.clean_X, poison])
        self.y = np.r_[self.clean_y, np.zeros(30, dtype=int)]

        t = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big"))
        tx = np.vstack([t.normal(c0, 1.0, (300, 2)), t.normal(c1, 1.0, (300, 2))])
        ty = np.r_[np.zeros(300), np.ones(300)].astype(int)
        self.accuracy = float((self.predict(tx)[0] == ty).mean())

    def predict(self, P):
        P = np.atleast_2d(P)
        D = np.linalg.norm(P[:, None, :] - self.X[None, :, :], axis=2)
        idx = np.argpartition(D, K, axis=1)[:, :K]
        d, yk = np.take_along_axis(D, idx, 1), self.y[idx]
        w = np.exp(-(d ** 2) / (2 * H ** 2))
        s1, s0 = (w * (yk == 1)).sum(1), (w * (yk == 0)).sum(1)
        p1 = (s1 + ALPHA) / (s0 + s1 + 2 * ALPHA)          # little evidence -> ~0.5
        return (p1 >= 0.5).astype(int), np.maximum(p1, 1 - p1)


@lru_cache(maxsize=4096)
def world(token):
    return World(token)


def flag_for(token):
    if not DYNAMIC_FLAGS:
        return STATIC_FLAG
    return "FLAG{island_" + hmac.new(SECRET, f"{token}|flag".encode(), hashlib.sha256).hexdigest()[:12] + "}"


# ---- identity: cookie (browser) or X-Token header (scripts) ----------------
@app.before_request
def identify():
    tok = request.headers.get("X-Token") or request.cookies.get("tok")
    g.new = False
    if not tok:
        tok, g.new = secrets.token_hex(8), True
    if not TOKEN_RE.match(tok):
        return jsonify(error="invalid token"), 400
    g.token = tok


@app.after_request
def set_cookie(resp):
    if getattr(g, "new", False):
        resp.set_cookie("tok", g.token, max_age=86400 * 7, samesite="Lax")
    return resp


def remaining(tok):
    return max(0, QUERY_BUDGET - used(tok))


def parse_points(raw):
    try:
        P = np.array(raw, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2 or not np.isfinite(P).all():
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("points must be a list of [x1, x2] numbers")
    if len(P) == 0 or len(P) > MAX_BATCH:
        raise ValueError(f"send between 1 and {MAX_BATCH} points per request")
    if np.abs(P).max() > LIMIT:
        raise ValueError(f"features must stay within [-{LIMIT:g}, {LIMIT:g}]")
    return P


@app.get("/")
def index():
    return send_from_directory("public", "index.html")


@app.get("/api/session")
def session():
    w = world(g.token)
    return jsonify(token=g.token, remaining=remaining(g.token), budget=QUERY_BUDGET, limit=LIMIT,
                   labels=LABELS, max_batch=MAX_BATCH, model_card={"heldout_accuracy": round(w.accuracy, 4)})


@app.get("/api/samples")
def samples():
    """Clean labelled reference rows: the baseline of 'normal' behaviour."""
    w = world(g.token)
    r = _rng(g.token, "samples")
    pick = np.r_[r.choice(200, 90, replace=False), 200 + r.choice(200, 90, replace=False)]
    return jsonify(points=np.round(w.clean_X[pick], 3).tolist(), labels=w.clean_y[pick].tolist())


@app.post("/api/predict")
def predict_one():
    body = request.get_json(silent=True) or {}
    return run([[body.get("x1"), body.get("x2")]], single=True)


@app.post("/api/batch")
def predict_batch():
    return run((request.get_json(silent=True) or {}).get("points"))


def run(raw, single=False):
    try:
        P = parse_points(raw)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not spend(g.token, len(P)):
        return jsonify(error="query budget exhausted", remaining=remaining(g.token)), 429
    lab, conf = world(g.token).predict(P)
    out = [{"label": LABELS[l], "confidence": round(float(c), 4)} for l, c in zip(lab, conf)]
    return jsonify(result=out[0] if single else out, remaining=remaining(g.token))


@app.post("/api/submit")
def submit():
    body = request.get_json(silent=True) or {}
    try:
        p = parse_points([[body.get("x1"), body.get("x2")]])[0]
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if too_many_wrong(g.token):
        return jsonify(error="too many wrong submissions, wait a minute"), 429
    ok = float(np.linalg.norm(p - world(g.token).trigger)) <= TOLERANCE
    if not ok:
        note_wrong(g.token)
    if ok:
        return jsonify(correct=True, flag=flag_for(g.token))
    return jsonify(correct=False, message="The model is not hiding anything at that point.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
