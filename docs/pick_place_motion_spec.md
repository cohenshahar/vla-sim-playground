# Pick-and-place motion spec — KR6 + vacuum (object → bin)

**Status**: DESIGN (numbers are starting points to calibrate, not validated).
**Owner**: Shahar Cohen · 2026-06-01 · learning/practice infra.
**Task**: lift an object off the table with the vacuum cup and place it inside an open bin.
**Pairs with**: `docs/vacuum_gripper_design.md` (adhesion + compliant cup) and the
`servo_to_pose` / `set_suction` helpers now in `kr6_ik_bridge.py`.

---

## 1. Scene changes needed

### 1a. The bin (this is the missing piece)
The scene has no container — `target_zone` is a flat pad. Add an **open box** (floor + 4
walls). Drop this into `includes/objects.xml`. Collision groups match the scene scheme
(world group 2/3) so both the arm and the object collide with it.

```xml
<!-- Open bin: inner cavity ~0.19 x 0.19 m, wall height ~0.08 m. Fixed body (no joint). -->
<!-- pos is on the table top (z~0.85); tune x,y to sit inside the arm's reach. -->
<body name="bin" pos="0.55 0.25 0.85">
  <geom name="bin_floor"  type="box" size="0.10 0.10 0.005" pos="0 0 0.005"
        rgba="0.55 0.38 0.20 1" contype="2" conaffinity="3"/>
  <geom name="bin_wall_px" type="box" size="0.005 0.10 0.04" pos="0.10 0 0.04"
        rgba="0.55 0.38 0.20 1" contype="2" conaffinity="3"/>
  <geom name="bin_wall_nx" type="box" size="0.005 0.10 0.04" pos="-0.10 0 0.04"
        rgba="0.55 0.38 0.20 1" contype="2" conaffinity="3"/>
  <geom name="bin_wall_py" type="box" size="0.10 0.005 0.04" pos="0 0.10 0.04"
        rgba="0.55 0.38 0.20 1" contype="2" conaffinity="3"/>
  <geom name="bin_wall_ny" type="box" size="0.10 0.005 0.04" pos="0 -0.10 0.04"
        rgba="0.55 0.38 0.20 1" contype="2" conaffinity="3"/>
  <site name="bin_center" pos="0 0 0.02" size="0.005" rgba="0 1 0 0.4"/>
</body>
```
Rim top ≈ bin.z + 0.08 = 0.93. Inner footprint = ±0.095 in x,y around the bin center.

### 1b. The object
Keep the existing free-body cube but it no longer needs to be a heavy "metal" box (we use
suction, not an electromagnet). Optional: rename `metal_box` → `target_object` (update any
`randomize_scene.py` / `demo_reach.py` references) and drop its mass to ~0.2 kg so the cup
holds it easily. The sphere stays out of v1 (suction can't seal a curved surface).

### 1c. Vacuum end-effector
Per `docs/vacuum_gripper_design.md`: replace `em_pad`/`em_weld` with the `suction_pad`
(compliant `slide` joint) + `act_vacuum` adhesion actuator, and set
`KR6_EE_SITE = "suction_tip_site"` in `kr6_ik_bridge.py`.

## 2. Reference heights (world Z)
Table top ≈ **0.85**. Cube center at rest ≈ 0.90, its top face ≈ **0.95**. Bin rim ≈ **0.93**.

| Waypoint | Z | Why |
|----------|----|-----|
| Hover above object | 1.05 | ~10 cm above the top face, safe pre-grasp |
| Seat target (descend to) | 0.94 | just *below* the top face so the compliant cup compresses; seat detection stops it |
| Transit / lift | 1.15 | clears the bin rim (0.93) with margin |
| Lower into bin | 1.00 | object bottom above bin floor; short drop on release |
| Retreat | 1.15 | clear out |

(X,Y come from the object pose for grasp waypoints and from `bin_center` for place waypoints.)

## 3. The motion (waypoint state machine)
Each step calls `bridge.servo_to_pose(target_pos, target_mat)` every tick until
`pos_err < tol`. Top-down grasp → `target_mat` = cup axis pointing world −Z (or use
position-only servo for the first bring-up and let the home pose keep the cup down).

```
0. HOME            -> known start joint config, vacuum OFF
1. HOVER           -> servo to (obj_x, obj_y, 1.05)            until pos_err < 5 mm
2. DESCEND/SEAT    -> servo toward (obj_x, obj_y, 0.94)        until SEATED (see §4)
3. ENGAGE          -> set_suction(on)  ; dwell ~8 ticks
4. LIFT            -> servo to (obj_x, obj_y, 1.15)            ; then CHECK GRASP (§4)
5. TRANSIT         -> servo to (bin_x,  bin_y,  1.15)
6. LOWER           -> servo to (bin_x,  bin_y,  1.00)
7. RELEASE         -> set_suction(off) ; dwell ~5 ticks
8. RETREAT         -> servo to (bin_x,  bin_y,  1.15)
9. HOME            -> return ; evaluate SUCCESS (§5)
```

## 4. Calibration thresholds (to tune on the strong PC)
- **Waypoint reached**: `pos_err < 0.005 m` (and `rot_err < 0.05 rad` if using orientation).
- **Seated** (stop descending, then engage): any of —
  `cup_compliance` joint qpos retracted > **0.002 m**, OR `sensor_ee_force` magnitude > **3 N**,
  OR `sensor_proximity` < **0.005 m**.
- **Suction dwell**: ~8 ticks after engage, ~5 after release.
- **Grasp confirmed** (after LIFT): object Z rose with the cup (object_z > rest_z + 0.03), else
  ABORT and retry/print failure. (This is exactly the hook the thesis Verifier would monitor.)

## 5. Success criterion (programmatic)
Object is "in the bin" when ALL hold at the end:
- `|obj_x − bin_x| < 0.09` and `|obj_y − bin_y| < 0.09` (inside inner footprint),
- `obj_z < 0.93` (below the rim),
- object linear speed < 0.02 m/s (at rest),
- vacuum OFF.

## 6. Randomization (for demos / robustness)
- Object initial X,Y sampled in the reachable table region (reuse `randomize_scene.py`),
  fixed flat orientation.
- Bin pose fixed for v1 (randomize later).
- Language instruction fixed: e.g. *"put the cube in the bin."*

## 7. Oracle skeleton (ties it together)
With `servo_to_pose` + `set_suction` already in `kr6_ik_bridge.py`, the oracle is small:

```python
# pick_place_oracle.py  (to write once the scene is updated)
WPTS_Z = dict(hover=1.05, seat=0.94, transit=1.15, lower=1.00)
def run(model, data, bridge, obj_xy, bin_xy):
    phase = "HOVER"; suction = False
    while not done:
        # pick target from phase, call bridge.servo_to_pose(...)
        # advance phase when pos_err < tol or SEATED; toggle set_suction at ENGAGE/RELEASE
        for _ in range(substeps): mujoco.mj_step(model, data)
    return success_check(data)   # §5
```

## 8. Why this is the backbone
This scripted oracle (a) validates scene + vacuum + compliance + IK end-to-end **before**
OpenVLA, (b) **generates the demos** for the LoRA fine-tune (the only path to KR6 success),
and (c) defines the **success check** used in eval. OpenVLA later replaces the waypoint logic
with `action_to_ctrl`; everything else (scene, suction, success check) is reused.

## 9. Integration checklist
- [ ] `objects.xml`: add the `bin` body; optionally rename object + reduce mass.
- [ ] `arm.xml` + `world.xml`: vacuum pad + adhesion, remove `em_weld` (vacuum design doc).
- [ ] `kr6_ik_bridge.py`: set `KR6_EE_SITE = "suction_tip_site"`. (`servo_to_pose`, `set_suction` done.)
- [ ] write `pick_place_oracle.py` from §7.
- [ ] validate on strong PC: scene loads → compliance test → suction lift test → full oracle → tune §4 numbers.

---

*vla-sim-playground / docs | Shahar Cohen | BGU Mechatronics | 2026-06-01 | DESIGN — calibrate before trusting numbers*
