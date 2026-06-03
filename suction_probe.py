"""Suction test v2: gentle, servo-like lift (small steps) so adhesion stays engaged.
Mimics servo_to_pose's max_cart_step motion rather than an abrupt ctrl jump."""
import numpy as np, mujoco

def make_xml(gain, stiffness, damping, margin, box_mass):
    return f"""
<mujoco model="suction_probe">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".3 .3 .3 1"/>
    <body name="box" pos="0 0 0.025">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.025 0.025 0.025" mass="{box_mass}"
            rgba="0.8 0.3 0.2 1" contype="1" conaffinity="1"/>
    </body>
    <body name="arm_mount" pos="0 0 0.30">
      <joint name="arm_z" type="slide" axis="0 0 1" range="-0.5 0.5" damping="50"/>
      <geom name="arm_geom" type="cylinder" fromto="0 0 0 0 0 0.05" size="0.01"
            rgba="0.2 0.2 0.8 1" contype="0" conaffinity="0"/>
      <body name="suction_pad" pos="0 0 0">
        <joint name="cup_compliance" type="slide" axis="0 0 -1"
               range="-0.010 0" stiffness="{stiffness}" damping="{damping}" springref="0"/>
        <geom name="cup_geom" type="cylinder" fromto="0 0 0 0 0 -0.005" size="0.03"
              rgba="0.15 0.15 0.18 1" margin="{margin}"
              solref="0.02 1" solimp="0.9 0.95 0.001" contype="1" conaffinity="1" mass="0.05"/>
        <site name="suction_tip_site" pos="0 0 -0.005" size="0.005" rgba="1 0 0 0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_arm_z" joint="arm_z" kp="3000" ctrlrange="-0.5 0.5"/>
    <adhesion name="act_vacuum" body="suction_pad" ctrlrange="0 1" gain="{gain}"/>
  </actuator>
</mujoco>"""

def ramp_arm(m, d, aid_arm, goal, rate=0.0008):
    """Move arm ctrl toward goal in small per-step increments (servo-like)."""
    while abs(d.ctrl[aid_arm] - goal) > 1e-4:
        step = np.clip(goal - d.ctrl[aid_arm], -rate, rate)
        d.ctrl[aid_arm] += step
        mujoco.mj_step(m, d)

def run(gain=40, stiffness=800, damping=5, margin=0.004, box_mass=0.5, verbose=True):
    m = mujoco.MjModel.from_xml_string(make_xml(gain, stiffness, damping, margin, box_mass))
    d = mujoco.MjData(m)
    aid_arm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_arm_z")
    aid_vac = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_vacuum")
    bid_box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    jadr_comp = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_compliance")]
    box_z = lambda: float(d.xpos[bid_box][2])

    mujoco.mj_forward(m, d)
    d.ctrl[aid_arm] = 0.0; d.ctrl[aid_vac] = 0.0
    for _ in range(300): mujoco.mj_step(m, d)
    rest = box_z()

    # DESCEND gently to seat (cup into box top)
    ramp_arm(m, d, aid_arm, -0.254, rate=0.0010)
    for _ in range(100): mujoco.mj_step(m, d)
    seat_comp = float(d.qpos[jadr_comp])

    # ENGAGE + dwell
    d.ctrl[aid_vac] = 1.0
    for _ in range(80): mujoco.mj_step(m, d)

    # LIFT gently (servo-like) to clear height
    ramp_arm(m, d, aid_arm, 0.10, rate=0.0008)
    for _ in range(200): mujoco.mj_step(m, d)
    lifted = box_z()
    grasped = lifted > rest + 0.05

    # RELEASE
    d.ctrl[aid_vac] = 0.0
    for _ in range(400): mujoco.mj_step(m, d)
    dropped = box_z()
    released = dropped < lifted - 0.05

    if verbose:
        print(f"gain={gain:>3} stiff={stiffness} margin={margin:<5} mass={box_mass}: "
              f"seat={seat_comp*1000:>5.1f}mm lifted={lifted:.3f} grasped={str(grasped):<5} "
              f"dropped={dropped:.3f} released={released}  {'OK' if grasped and released else 'FAIL'}")
    return dict(grasped=grasped, released=released, lifted=lifted, dropped=dropped)

if __name__ == "__main__":
    print("=== gentle lift, sweep margin (gain=40, mass=0.5) ===")
    for mg in (0.004, 0.010, 0.020, 0.030):
        run(gain=40, margin=mg, box_mass=0.5)
    print("=== gentle lift, sweep gain (margin=0.02, mass=0.5) ===")
    for g in (10, 20, 40):
        run(gain=g, margin=0.020, box_mass=0.5)
    print("=== lighter box 0.2 kg (margin=0.02, gain=40) ===")
    run(gain=40, margin=0.020, box_mass=0.2)

def lift_rate_study():
    print("\n=== lift-speed vs margin (gain=40, mass=0.5): does fast lift break the grab? ===")
    print("rate is arm motion per physics step (proxy for servo max_cart_step / substeps)")
    for margin in (0.004, 0.010, 0.020):
        for rate in (0.0008, 0.002, 0.004, 0.008):
            m = mujoco.MjModel.from_xml_string(make_xml(40, 800, 5, margin, 0.5))
            d = mujoco.MjData(m)
            aid_arm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_arm_z")
            aid_vac = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_vacuum")
            bid_box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
            mujoco.mj_forward(m, d)
            for _ in range(300): mujoco.mj_step(m, d)
            rest = float(d.xpos[bid_box][2])
            ramp_arm(m, d, aid_arm, -0.254, rate=0.001)
            for _ in range(100): mujoco.mj_step(m, d)
            d.ctrl[aid_vac] = 1.0
            for _ in range(80): mujoco.mj_step(m, d)
            ramp_arm(m, d, aid_arm, 0.10, rate=rate)
            for _ in range(200): mujoco.mj_step(m, d)
            lifted = float(d.xpos[bid_box][2])
            ok = lifted > rest + 0.05
            print(f"  margin={margin:<5} rate={rate:<6}: lifted={lifted:.3f} {'HOLD' if ok else 'DROP'}")

lift_rate_study()
