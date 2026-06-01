# Vacuum gripper — MuJoCo design note

**Status**: DESIGN (not built, not tested). Draft to validate on the strong PC / Ubuntu.
**Owner**: Shahar Cohen · 2026-06-01 · learning/practice infra (not thesis content).
**Replaces**: the EM (electromagnet) end-effector — `em_pad`, `em_weld`, `em_controller`.
**Targets**: the KR6 scene in `VLA_Tutorial/VLATraining/sim` (`includes/arm.xml`,
`scene/world.xml`, `includes/objects.xml`).

---

## 1. Goal

Model the new **vacuum / suction arm** in MuJoCo, including the **passive compliant cup**
Shahar specified: when the tool touches an object it can compress a few mm inward, so the
rigid arm never drives *into* the object. Two independent pieces:

1. **Suction** = an attractive contact force that holds an object to the cup while active.
2. **Passive compliance** = a short spring-loaded travel along the tool axis.

These are separate MuJoCo features and are designed independently below.

## 2. Why the compliance matters (the physics)

The KR6 joints are **position-controlled** (`act_a1..a6`, target angles). Position control is
stiff: if the commanded pose puts the rigid cup 3 mm *past* the object surface, the controller
will push hard to reach that target, producing huge contact forces, interpenetration, or a
blown-up sim step. A real suction cup has a compliant bellows/spring that absorbs that
approach error so the cup seats flush. We reproduce it with a sprung prismatic joint:

- It **absorbs approach/pose error** (the IK and OpenVLA won't be mm-accurate).
- It gives the grasp controller a **sensable contact signal** (slide displacement or EE force)
  to decide *when* to switch suction on.
- It keeps the contact **soft and stable** for the physics solver.

## 3. Piece A — suction via an `adhesion` actuator

MuJoCo (≥2.3, and your 3.1.6) has a built-in **`adhesion`** actuator built exactly for
suction/vacuum grippers. Semantics:

- It is attached to a **body**; it applies an attractive force to geoms currently **in contact**
  with that body's geoms, pulling them toward the gripper.
- Max total force = `gain * ctrl`, with `ctrl` in `[0, 1]`. Distributed across active contacts.
- It only acts when there **is** contact (within the geom's `margin`). So adding a small
  `margin` to the cup geom lets suction "grab" right as the cup approaches/seats.

So `a[6]` (OpenVLA gripper, 0=open/1=closed) → adhesion `ctrl` (0 = vacuum off, 1 = full
vacuum). No weld, no manual relpose bookkeeping — this is the physically meaningful model and
it replaces the entire `em_weld` + `em_controller.em_activate()` machinery.

```xml
<!-- in <actuator> of arm.xml, alongside act_a1..a6 -->
<adhesion name="act_vacuum" body="suction_pad" ctrlrange="0 1" gain="40"/>
<!-- gain=40 N max hold force: > m*g for the 0.5 kg box (≈5 N) with margin for dynamics.
     TUNE: too low -> object slips on lift; too high -> object snaps unrealistically. -->
```

## 4. Piece B — passive compliance via a sprung `slide` joint

Insert a **prismatic (`slide`) joint** between `link_6` and a new `suction_pad` body, along the
tool approach axis (local **+X** in your arm, same axis the old `em_pad` used). Make it passive:
no actuator, just `stiffness` (spring back to the extended rest pose) and `damping`.

```xml
<!-- replaces the old <body name="em_pad" ...> block under link_6 -->
<body name="suction_mount" pos="0.01 0 0">
  <!-- passive compliant travel along +X (tool axis). Rest = fully extended (0).
       range lets it retract up to 10 mm under contact load. -->
  <joint name="cup_compliance" type="slide" axis="1 0 0"
         range="-0.010 0" stiffness="800" damping="5" springref="0"/>

  <body name="suction_pad">
    <!-- the cup: soft contact so it conforms, small margin so adhesion engages on approach -->
    <geom name="cup_geom" type="cylinder" fromto="-0.005 0 0  0.005 0 0" size="0.03"
          rgba="0.15 0.15 0.18 1" margin="0.004"
          solref="0.02 1" solimp="0.9 0.95 0.001"
          contype="1" conaffinity="6"/>
    <!-- NEW end-effector reference site = cup contact face (IK targets THIS) -->
    <site name="suction_tip_site" pos="0.006 0 0" euler="0 -90 0" size="0.005" rgba="1 0 0 0.5"/>
    <!-- keep the wrist camera on the moving pad -->
    <camera name="cam_wrist" pos="-0.02 0 0.10" xyaxes="0 1 0 -0.342 0 -0.940" fovy="80"/>
    <!-- contact/force sensing site for the grasp controller -->
    <site name="cup_contact_site" pos="0.006 0 0" size="0.004" rgba="1 0.5 0 0"/>
  </body>
</body>
```

Tuning intent: `stiffness=800 N/m` gives ≈6 mm deflection under a ~5 N approach load; raise it
if the cup feels mushy, lower it if the arm still spikes contact force. `damping` kills bounce.
`range=-10..0 mm` caps the travel. `solref/solimp` soften the cup contact so it seats instead
of chattering.

## 5. Grasp controller (replaces `em_controller.py`)

State machine, driven by the sensors you already have (`sensor_proximity` rangefinder,
`sensor_ee_force`, plus the new `cup_compliance` joint position):

1. **Approach** — IK drives the tip toward the object; vacuum off (`ctrl=0`).
2. **Seat** — when `cup_compliance` has retracted past a small threshold (e.g. > 2 mm) **or**
   `sensor_ee_force` exceeds a few N → the cup is touching.
3. **Engage** — set `act_vacuum` ctrl = 1. Hold a few steps to let adhesion lock.
4. **Lift / move** — object now follows the cup.
5. **Release** — `act_vacuum` ctrl = 0.

For OpenVLA-driven runs, `a[6] > 0.5` requests "closed"; gate the actual engage on the seat
condition so it can't "grab air." This is the suction analogue of the old proximity check.

## 6. Sensors — reuse, re-point

Keep the Phase 7 sensor suite; just move the EE-referenced ones from `em_contact_site`/
`em_touch_site` to the new `cup_contact_site` / `suction_tip_site`. The rangefinder (proximity)
and force/torque sensors become the *trigger signals* for the grasp state machine above.

## 7. Impact on `kr6_ik_bridge.py` (small)

- **EE site**: change `KR6_EE_SITE = "em_contact_site"` → `"suction_tip_site"` (the cup face).
  The IK now targets where suction actually contacts.
- **Passive joint is invisible to IK**: `cup_compliance` is unactuated, so it is *not* in
  `arm_joints`/`arm_actuators`. The IK still solves the 6 arm joints only. The bridge reads the
  tip site pose, which already includes any compression — correct by construction.
- **Gripper helper**: retire `set_weld_active(...)`; add
  `set_suction(model, data, "act_vacuum", on)` that writes the adhesion actuator's ctrl
  (`data.ctrl[act_id] = 1.0 if on else 0.0`). Wire `diag["gripper_closed"]` to it (gated by the
  seat condition in §5).

## 8. EM-removal touchpoints (what to find & change)

Before this lands, grep the KR6 codebase for EM references and migrate each:
- `scene/world.xml` — remove the `<equality><weld name="em_weld" .../></equality>` block.
- `includes/arm.xml` — replace the `em_pad` body (done in §4); drop `em_touch_geom`,
  `em_*` sites.
- `em_controller.py` — replace with the suction state machine (§5).
- any `randomize_scene.py` / `demo_reach.py` references to `em_*` names.
- `demo_reach.py` (Phase 10.1) only *hovers* using proximity — it keeps working, but update the
  site name it reads.

## 9. Validation plan (on the strong PC / Ubuntu, in order)

1. **Loads**: `mujoco.MjModel.from_xml_path(world.xml)` succeeds; check `act_vacuum` appears in
   actuators and `cup_compliance` in joints (`model.nu`, `model.nq` increased by 1 each).
2. **Compliance**: command the arm straight down onto the box with vacuum off — the cup should
   retract a few mm and contact force stay bounded (no blow-up). Plot `cup_compliance` qpos.
3. **Suction**: with the cup seated, set `act_vacuum=1`, lift the arm — the box should follow;
   set `act_vacuum=0` — it should drop. This is the suction analogue of the old EM weld test.
4. **Tune** `gain`, `stiffness`, `damping`, `margin` per the notes above.
5. Only then wire it under OpenVLA via the updated `kr6_ik_bridge` + `run_openvla_kr6`.

## 10. Open decisions (your call)

- **Adhesion vs weld-on-contact**: `adhesion` is the physically correct suction model and is
  recommended. A fallback is the old approach (activate a weld when seated) if adhesion proves
  finicky to tune — keep it in your back pocket, don't start there.
- **Axial-only vs lateral compliance**: this design models axial give only (one slide joint).
  Real cups also flex sideways. Start axial-only; add a 2-DoF gimbal/ball later only if grasps
  fail on tilted surfaces.
- **Cup as single geom vs cup + lip**: single cylinder geom here for simplicity; a separate soft
  "lip" geom is a later refinement for nicer seating on edges.
- **Where the EE site sits**: at the cup face (chosen) vs a few mm ahead (pre-contact). Face is
  the right IK target for grasping.

---

*vla-sim-playground / docs | Shahar Cohen | BGU Mechatronics | 2026-06-01 | DESIGN — validate before trusting numbers*
