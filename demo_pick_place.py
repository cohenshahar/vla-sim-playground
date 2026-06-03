#!/usr/bin/env python3
"""
demo_pick_place.py  --  KR6 suction pick-and-place, random reachable scenarios.

WATCH a single run:
    conda deactivate
    python3 demo_pick_place.py            # random scenario
    python3 demo_pick_place.py 7          # reproducible (seed 7)

MEASURE the oracle (GATE 1) -- headless, fast, no window:
    python3 demo_pick_place.py --batch 30        # 30 random scenarios
    python3 demo_pick_place.py --batch 30 100    # 30 scenarios starting at seed 100

Box/crate are drawn from a precomputed cup-down reachability map
(cached in reachable_cupdown.npz; delete it to rebuild).
"""
import sys, os, time
import numpy as np
import mujoco

# ----------------------------- TUNABLE KNOBS --------------------------------
N_SAMPLE   = 150000
TILT_MAX   = 12.0   # relaxed: cup 'roughly down', not perfectly vertical
GRASP_Z    = (0.95, 1.05)
REGION_X   = (0.22, 0.52)   # widened; grab is now descend-to-contact
REGION_Y   = (-0.26, 0.15)   # widened
MIN_SEP    = 0.24
CRATE_FIT_X = (-0.70, 0.70)
CRATE_FIT_Y = (-0.24, 0.24)
PATH_TILT_WARN = 20.0

ARM_DAMPING = 40.0; COMP_DAMPING = 15.0; VAC_GAIN = 100.0
VAC_ON_STEPS = 200; VAC_OFF_STEPS = 300; LIFT_STEPS = 800
SEGMENT_STEPS = 300; N_CARRY = 10
SEAT_Z = 0.948; RELEASE_GAP = 0.012
CACHE = "reachable_cupdown.npz"
# ---------------------------------------------------------------------------

ARM_JOINTS = [f"joint_a{i}" for i in range(1, 7)]
ARM_ACTS   = [f"act_a{i}"   for i in range(1, 7)]
m = mujoco.MjModel.from_xml_path("scene/world.xml")
d = mujoco.MjData(m)
dk = mujoco.MjData(m)   # scratch for IK/queries; never touches the live sim
def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def aid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
def bidn(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "suction_tip_site")
qadr   = [m.jnt_qposadr[jid(j)] for j in ARM_JOINTS]
dofadr = [int(m.jnt_dofadr[jid(j)]) for j in ARM_JOINTS]
act    = [aid(a) for a in ARM_ACTS]
lo = np.array([m.jnt_range[jid(j)][0] for j in ARM_JOINTS])
hi = np.array([m.jnt_range[jid(j)][1] for j in ARM_JOINTS])
vac    = aid("act_vacuum")
box    = bidn("metal_box"); binb = bidn("bin")
boxq   = int(m.jnt_qposadr[int(m.body_jntadr[box])])
boxdof = int(m.body_dofadr[box])
cjnt   = jid("cup_compliance")
cq     = int(m.jnt_qposadr[cjnt]); cdof = int(m.jnt_dofadr[cjnt])

Rps = np.zeros(9); mujoco.mju_quat2Mat(Rps, m.site_quat[sid]); Rps = Rps.reshape(3, 3)
Rsite_des = np.column_stack([[0,0,-1.0], [1.0,0,0], np.cross([0,0,-1.0],[1.0,0,0])]) @ Rps

# configure model once
for dd in dofadr: m.dof_damping[dd] = ARM_DAMPING
m.dof_damping[cdof] = COMP_DAMPING
m.actuator_gainprm[vac][0] = VAC_GAIN
def set_geom(n, size, pos):
    g = gid(n); m.geom_size[g] = size; m.geom_pos[g] = pos
set_geom("bin_floor",   [0.16, 0.16, 0.005], [0, 0, 0.005])
set_geom("bin_wall_px", [0.005, 0.16, 0.03], [ 0.16, 0, 0.03])
set_geom("bin_wall_nx", [0.005, 0.16, 0.03], [-0.16, 0, 0.03])
set_geom("bin_wall_py", [0.16, 0.005, 0.03], [0,  0.16, 0.03])
set_geom("bin_wall_ny", [0.16, 0.005, 0.03], [0, -0.16, 0.03])
FLOOR_TOP = 0.85 + 0.005 + 0.005
REST_Z    = FLOOR_TOP + RELEASE_GAP + 0.10 + 0.005

def tilt_of(q):
    dk.qpos[qadr] = q; mujoco.mj_forward(m, dk)
    face = (np.array(dk.site_xmat[sid]).reshape(3,3) @ Rps.T) @ np.array([1,0,0])
    return np.degrees(np.arccos(np.clip(face @ np.array([0,0,-1.0]), -1, 1)))

def path_tilt(waypts, sub=8):
    mx = 0.0
    for A, B in zip(waypts[:-1], waypts[1:]):
        for t in np.linspace(0, 1, sub):
            mx = max(mx, tilt_of((1-t)*np.array(A) + t*np.array(B)))
    return mx

def ik(pos, q0, iters=500):
    q = np.array(q0, float)
    for _ in range(iters):
        dk.qpos[qadr] = q; mujoco.mj_forward(m, dk)
        p = np.array(dk.site_xpos[sid]); R = np.array(dk.site_xmat[sid]).reshape(3, 3)
        Rerr = Rsite_des @ R.T; q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, Rerr.reshape(9))
        ang = 2*np.arccos(np.clip(q4[0], -1, 1)); v = q4[1:4]; nv = np.linalg.norm(v)
        rot = (v/nv*ang) if nv > 1e-9 else np.zeros(3)
        err = np.concatenate([pos - p, rot])
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, dk, jp, jr, sid)
        J = np.vstack([jp[:, dofadr], jr[:, dofadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.03*np.eye(6), err)
        q = np.clip(q + np.clip(dq, -0.15, 0.15), lo, hi)
        if np.linalg.norm(pos - p) < 0.0025 and abs(ang) < 0.03: break
    return q

# --------------------- reachability map (build or load) ---------------------
def build_map():
    print(f"building cup-down reachability map ({N_SAMPLE} samples, one time)...")
    rng = np.random.default_rng(0); xy = []; qs = []
    for _ in range(N_SAMPLE):
        q = rng.uniform(lo, hi); dk.qpos[qadr] = q; mujoco.mj_forward(m, dk)
        face = (np.array(dk.site_xmat[sid]).reshape(3,3) @ Rps.T) @ np.array([1,0,0])
        if face[2] > -np.cos(np.radians(TILT_MAX)): continue
        ee = np.array(dk.site_xpos[sid])
        if (REGION_X[0] < ee[0] < REGION_X[1] and REGION_Y[0] < ee[1] < REGION_Y[1]
                and GRASP_Z[0] < ee[2] < GRASP_Z[1]):
            xy.append(ee[:2].copy()); qs.append(q.copy())
    xy = np.array(xy); qs = np.array(qs); np.savez(CACHE, xy=xy, qs=qs, tilt_max=TILT_MAX)
    print(f"  -> {len(xy)} reachable cup-down spots")
    return xy, qs
if os.path.exists(CACHE):
    z = np.load(CACHE)
    if float(z["tilt_max"]) == TILT_MAX and len(z["xy"]) > 1:
        map_xy, map_q = z["xy"], z["qs"]
    else:
        map_xy, map_q = build_map()
else:
    map_xy, map_q = build_map()
if len(map_xy) < 2:
    raise SystemExit("reachability map too small; widen REGION_* or raise TILT_MAX")
FITS = [i for i in range(len(map_xy))
        if CRATE_FIT_X[0] < map_xy[i][0] < CRATE_FIT_X[1]
        and CRATE_FIT_Y[0] < map_xy[i][1] < CRATE_FIT_Y[1]]

# --------------------------- scenario + planning ----------------------------
def choose_scenario(seed):
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        ci = FITS[rng.integers(len(FITS))]
        cand = [i for i in range(len(map_xy)) if np.linalg.norm(map_xy[i]-map_xy[ci]) > MIN_SEP]
        if cand:
            bi = cand[rng.integers(len(cand))]
            # random valid arm start
            qs = None
            for _ in range(20000):
                q = rng.uniform(lo, hi); dk.qpos[qadr] = q; mujoco.mj_forward(m, dk)
                if dk.site_xpos[sid][2] >= 1.00 and dk.ncon == 0: qs = q; break
            return dict(PX=map_xy[bi][0], PY=map_xy[bi][1], BX=map_xy[ci][0], BY=map_xy[ci][1],
                        SEED_PICK=map_q[bi], SEED_PLACE=map_q[ci],
                        q_start=qs if qs is not None else map_q[bi])
    return None

def plan(s):
    PX, PY, BX, BY = s["PX"], s["PY"], s["BX"], s["BY"]
    s["q_home"]    = ik([0.28, -0.05, 1.22], s["SEED_PICK"])
    s["q_hi_pick"] = ik([PX, PY, 1.14], s["SEED_PICK"])
    s["q_seat"]    = ik([PX, PY, SEAT_Z], s["q_hi_pick"])
    carry = [s["q_hi_pick"]]; seed = s["q_hi_pick"]
    for a in np.linspace(0.0, 1.0, N_CARRY)[1:]:
        seed = ik([PX + a*(BX-PX), PY + a*(BY-PY), 1.16], seed); carry.append(seed)
    s["carry"] = carry; s["q_hi_place"] = carry[-1]
    s["q_rest"]  = ik([BX, BY, REST_Z], carry[-1])
    s["q_clear"] = ik([BX, BY, 1.08], s["q_rest"])
    path = [s["q_seat"]] + carry + [s["q_rest"], s["q_clear"]]
    s["maxstep"] = max(float(np.max(np.abs(np.array(b)-np.array(a)))) for a,b in zip(path[:-1],path[1:]))
    s["carry_tilt"] = path_tilt(s["carry"] + [s["q_rest"]])
    s["seat_tilt"]  = tilt_of(s["q_seat"]); s["crate_tilt"] = tilt_of(s["q_rest"])
    return s

# ------------------------------ execution -----------------------------------
def run_episode(s, viewer=False, realtime=False, verbose=False):
    PX, PY, BX, BY = s["PX"], s["PY"], s["BX"], s["BY"]
    m.body_pos[binb] = [BX, BY, 0.85]
    mujoco.mj_resetData(m, d)
    d.qpos[qadr] = s["q_start"]; d.ctrl[act] = s["q_start"]
    d.qpos[boxq:boxq+3] = [PX, PY, 0.90]; d.qpos[boxq+3:boxq+7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    boxp = lambda: np.array(d.xpos[box]); boxv = lambda: float(np.linalg.norm(d.qvel[boxdof:boxdof+3]))
    V = None
    if viewer:
        from mujoco import viewer as _vmod
        V = _vmod.launch_passive(m, d)
        V.cam.lookat[:] = [(PX+BX)/2, (PY+BY)/2, 0.95]
        V.cam.distance = 1.8; V.cam.azimuth = 135; V.cam.elevation = -22; V.sync()
    def gcs():
        mujoco.mj_step1(m, d); d.qfrc_applied[dofadr] = d.qfrc_bias[dofadr]; mujoco.mj_step2(m, d)
        if V is not None:
            V.sync()
            if realtime: time.sleep(m.opt.timestep)
    def move(qA, qB, n):
        for i in range(n): d.ctrl[act] = (1-(i+1)/n)*np.array(qA)+((i+1)/n)*np.array(qB); gcs()
    def ramp(a, b, n):
        for i in range(n): d.ctrl[vac] = a+(b-a)*(i+1)/n; gcs()
    def settle(n):
        for _ in range(n): gcs()
    def rep(tag):
        if verbose: print(f"{tag:10s} box={np.round(boxp(),3).tolist()}  v={boxv():.3f}  comp={d.qpos[cq]*1000:6.1f}mm")
    settle(60); rep("start")
    move(s["q_start"], s["q_home"], 500)
    move(s["q_home"], s["q_hi_pick"], 500); rep("reached")
    A = s["q_hi_pick"]; B = s["q_seat"]          # final straight-down descent...
    for i in range(400):                          # ...stop the instant the cup contacts the box
        a = (i+1)/400; d.ctrl[act] = (1-a)*np.array(A) + a*np.array(B); gcs()
        if abs(d.qpos[cq]) > 0.003: break
    settle(60); rep("seated")
    ramp(0.0, VAC_GAIN, VAC_ON_STEPS); settle(120); rep("vacuum-on")
    move(s["q_seat"], s["q_hi_pick"], LIFT_STEPS); settle(150); rep("lifted")
    for A, B in zip(s["carry"][:-1], s["carry"][1:]): move(A, B, SEGMENT_STEPS)
    settle(120); rep("carried")
    # the box hangs off-centre from the cup; shift the cup so the BOX centres on the crate
    ee_xy = np.array(d.site_xpos[sid])[:2]; bxy = np.array(d.xpos[box])[:2]
    off = bxy - ee_xy
    q_rest  = ik([BX - off[0], BY - off[1], REST_Z], s["q_hi_place"])
    q_clear = ik([BX - off[0], BY - off[1], 1.08], q_rest)
    if verbose: print(f"           box hang offset = {np.round(off,3).tolist()} -> re-centring")
    move(s["q_hi_place"], q_rest, 500); settle(150); rep("over-crate")
    ramp(VAC_GAIN, 0.0, VAC_OFF_STEPS); rep("released"); settle(150)
    move(q_rest, q_clear, 250); settle(300); rep("clear")
    bp = boxp(); binp = np.array(d.xpos[binb]); v = boxv()
    ok = (abs(bp[0]-binp[0]) < 0.14 and abs(bp[1]-binp[1]) < 0.14 and 0.88 < bp[2] < 0.93 and v < 0.03)
    if V is not None:
        print("\n(viewer left open -- close the window to exit)")
        while V.is_running(): time.sleep(0.1)
    return bool(ok), bp, v

# --------------------------------- main -------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        N = int(sys.argv[2]); start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        print(f"\nGATE 1 batch: {N} scenarios (seeds {start}..{start+N-1})\n")
        n_ok = n_hard = n_hard_ok = 0; fails = []
        for k in range(N):
            seed = start + k
            s = choose_scenario(seed)
            if s is None: print(f"seed {seed}: no scenario"); continue
            plan(s)
            ok, bp, v = run_episode(s, viewer=False, realtime=False, verbose=False)
            hard = s["carry_tilt"] > PATH_TILT_WARN
            n_ok += ok; n_hard += hard; n_hard_ok += (hard and ok)
            flag = "HARD" if hard else "    "
            miss = np.hypot(bp[0]-s['BX'], bp[1]-s['BY'])
            print(f"seed {seed:4d} {flag}  sep={np.hypot(s['PX']-s['BX'],s['PY']-s['BY']):.2f} "
                  f"carry_tilt={s['carry_tilt']:4.1f}  -> {'OK ' if ok else 'FAIL'}  "
                  f"pick=[{s['PX']:.2f},{s['PY']:.2f}] crate=[{s['BX']:.2f},{s['BY']:.2f}] "
                  f"box=[{bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}] miss={miss:.2f} "
                  f"z={'in' if 0.86<bp[2]<0.93 else 'OUT'}")
            if not ok: fails.append(seed)
        print(f"\n--- RESULT ---")
        print(f"overall : {n_ok}/{N} = {100*n_ok/N:.0f}%")
        easy = N - n_hard
        if easy: print(f"easy    : {n_ok-n_hard_ok}/{easy} = {100*(n_ok-n_hard_ok)/easy:.0f}%  (carry_tilt<={PATH_TILT_WARN:.0f})")
        if n_hard: print(f"hard    : {n_hard_ok}/{n_hard} = {100*n_hard_ok/n_hard:.0f}%  (carry_tilt>{PATH_TILT_WARN:.0f})")
        print(f"GATE 1 (>=90%): {'PASS' if n_ok/N>=0.90 else 'not yet'}")
        if fails: print(f"failing seeds: {fails}")
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(1, 100000)
        print(f"\n=== scenario seed = {seed} ===")
        s = choose_scenario(seed)
        if s is None: raise SystemExit("could not build a scenario; lower MIN_SEP")
        plan(s)
        print(f"box at ({s['PX']:.3f},{s['PY']:.3f})  crate at ({s['BX']:.3f},{s['BY']:.3f})  "
              f"sep={np.hypot(s['PX']-s['BX'],s['PY']-s['BY']):.2f}")
        print(f"seat tilt={s['seat_tilt']:.1f}  crate tilt={s['crate_tilt']:.1f}  "
              f"max carry tilt={s['carry_tilt']:.1f}  largest joint step={s['maxstep']:.2f} rad")
        if s["carry_tilt"] > PATH_TILT_WARN:
            print(f"** WARNING: carry tilts up to {s['carry_tilt']:.1f} deg -- hard scenario, box may slip **")
        ok, bp, v = run_episode(s, viewer=True, realtime=True, verbose=True)
        print(f"\nFINAL box={np.round(bp,3).tolist()}  v={v:.4f}\nSUCCESS = {ok}  (seed {seed})")
