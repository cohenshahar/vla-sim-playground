"""
kr6_ik_bridge.py
================
Map an OpenVLA 7-D end-effector action onto the KUKA KR6 R900's six *position*
actuators via damped-least-squares (DLS) inverse kinematics, and provide an
absolute-pose servo for scripted / oracle waypoint motion (pick-and-place).

OpenVLA action (BridgeData-v2):
    a[0:3] dEE position (EE frame), a[3:6] dEE orientation (axis-angle),
    a[6] gripper (0=open, 1=closed).

KR6 scene exposes 6 position actuators act_a1..a6 (target joint angles, rad) and
an EE site (em_contact_site -> suction_tip_site for the vacuum cup).

    action_to_ctrl(a)       : OpenVLA delta -> joint targets (policy path)
    servo_to_pose(pos, mat) : absolute pose -> joint targets (oracle/waypoint path)

STATUS: UNTESTED ON HARDWARE. Validate on the strong PC; tune pos_step, rot_step,
damping, max_joint_step on first run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


KR6_EE_SITE = "suction_tip_site"   # vacuum cup face (was "em_contact_site" for the EM gripper)
KR6_ARM_JOINTS = ("joint_a1", "joint_a2", "joint_a3",
                  "joint_a4", "joint_a5", "joint_a6")
KR6_ARM_ACTUATORS = ("act_a1", "act_a2", "act_a3",
                     "act_a4", "act_a5", "act_a6")
KR6_EM_WELD = "em_weld"           # legacy EM grip (replaced by suction)
KR6_VACUUM_ACT = "act_vacuum"     # adhesion actuator for the vacuum cup


@dataclass
class KR6IKBridge:
    """DLS IK bridge: OpenVLA action (or absolute pose) -> KR6 joint targets."""

    model: object
    data: object
    ee_site: str = KR6_EE_SITE
    arm_joints: Sequence[str] = KR6_ARM_JOINTS
    arm_actuators: Sequence[str] = KR6_ARM_ACTUATORS
    pos_step: float = 0.02
    rot_step: float = 0.04
    damping: float = 0.1
    max_joint_step: float = 0.10
    use_orientation: bool = True
    gripper_threshold: float = 0.5

    _site_id: int = field(default=-1, init=False)
    _dof_adr: np.ndarray = field(default=None, init=False)
    _qpos_adr: np.ndarray = field(default=None, init=False)
    _act_ids: np.ndarray = field(default=None, init=False)
    _jnt_range: np.ndarray = field(default=None, init=False)

    def __post_init__(self) -> None:
        import mujoco

        m = self.model
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        if self._site_id < 0:
            raise ValueError(f"EE site {self.ee_site!r} not found in model.")

        dof_adr, qpos_adr, jnt_range = [], [], []
        for jname in self.arm_joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"joint {jname!r} not found in model.")
            dof_adr.append(int(m.jnt_dofadr[jid]))
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            jnt_range.append(m.jnt_range[jid].copy())
        self._dof_adr = np.asarray(dof_adr, dtype=int)
        self._qpos_adr = np.asarray(qpos_adr, dtype=int)
        self._jnt_range = np.asarray(jnt_range, dtype=float)

        act_ids = []
        for aname in self.arm_actuators:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"actuator {aname!r} not found in model.")
            act_ids.append(int(aid))
        self._act_ids = np.asarray(act_ids, dtype=int)

        if len(self.arm_actuators) != len(self.arm_joints):
            raise ValueError("arm_joints and arm_actuators must be the same length.")

    def _dls_dq(self, J, err):
        """Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err."""
        lam2 = self.damping ** 2
        JJt = J @ J.T
        A = JJt + lam2 * np.eye(JJt.shape[0])
        return J.T @ np.linalg.solve(A, err)

    def action_to_ctrl(self, action: np.ndarray) -> dict:
        """OpenVLA 7-D action -> KR6 joint targets (applied to data.ctrl)."""
        import mujoco

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != 7:
            raise ValueError(f"expected 7-D action, got shape={a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action: {a.tolist()}")

        m, d = self.model, self.data
        ee_pos = np.array(d.site_xpos[self._site_id], dtype=float)
        ee_mat = np.array(d.site_xmat[self._site_id], dtype=float).reshape(3, 3)

        dpos_world = ee_mat @ (self.pos_step * a[0:3])
        if self.use_orientation:
            drot_world = ee_mat @ (self.rot_step * a[3:6])
            task_err = np.concatenate([dpos_world, drot_world])
        else:
            task_err = dpos_world

        jacp = np.zeros((3, m.nv), dtype=float)
        jacr = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacSite(m, d, jacp, jacr, self._site_id)
        Jp = jacp[:, self._dof_adr]
        J = np.vstack([Jp, jacr[:, self._dof_adr]]) if self.use_orientation else Jp

        dq = self._dls_dq(J, task_err)
        np.clip(dq, -self.max_joint_step, self.max_joint_step, out=dq)

        q_cur = d.qpos[self._qpos_adr].copy()
        q_target = q_cur + dq
        lo, hi = self._jnt_range[:, 0], self._jnt_range[:, 1]
        has_range = hi > lo
        q_target = np.where(has_range, np.clip(q_target, lo, hi), q_target)
        d.ctrl[self._act_ids] = q_target

        gripper_raw = float(a[6])
        return {
            "q_cur": q_cur, "q_target": q_target, "dq": dq, "ee_pos": ee_pos,
            "task_err": task_err, "gripper_raw": gripper_raw,
            "gripper_closed": gripper_raw > self.gripper_threshold,
        }

    def servo_to_pose(self, target_pos, target_mat=None, *, pos_gain=2.0,
                      rot_gain=1.0, max_cart_step=0.02) -> dict:
        """Absolute target pose -> KR6 joint targets, one control tick.

        Engine for scripted / oracle waypoint motion. target_mat=None => position
        only (robust first pass). Call each tick until pos_err (and rot_err) < tol.
        Returns {"pos_err"(m), "rot_err"(rad), "q_target"(6,)}.
        """
        import mujoco

        m, d = self.model, self.data
        cur_pos = np.array(d.site_xpos[self._site_id], dtype=float)
        cur_mat = np.array(d.site_xmat[self._site_id], dtype=float).reshape(3, 3)

        pos_err_vec = np.asarray(target_pos, dtype=float) - cur_pos
        pos_task = pos_gain * pos_err_vec
        n = np.linalg.norm(pos_task)
        if n > max_cart_step:
            pos_task = pos_task / n * max_cart_step

        jacp = np.zeros((3, m.nv), dtype=float)
        jacr = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacSite(m, d, jacp, jacr, self._site_id)
        Jp = jacp[:, self._dof_adr]

        rot_angle = 0.0
        if target_mat is not None:
            R_err = np.asarray(target_mat, dtype=float) @ cur_mat.T
            quat = np.zeros(4, dtype=float)
            mujoco.mju_mat2quat(quat, R_err.reshape(9))
            w = float(np.clip(quat[0], -1.0, 1.0))
            rot_angle = 2.0 * np.arccos(w)
            v = quat[1:4]
            nv = np.linalg.norm(v)
            rot_vec = (v / nv) * rot_angle if nv > 1e-9 else np.zeros(3)
            task = np.concatenate([pos_task, rot_gain * rot_vec])
            J = np.vstack([Jp, jacr[:, self._dof_adr]])
        else:
            task = pos_task
            J = Jp

        dq = self._dls_dq(J, task)
        np.clip(dq, -self.max_joint_step, self.max_joint_step, out=dq)

        q_cur = d.qpos[self._qpos_adr].copy()
        q_target = q_cur + dq
        lo, hi = self._jnt_range[:, 0], self._jnt_range[:, 1]
        has_range = hi > lo
        q_target = np.where(has_range, np.clip(q_target, lo, hi), q_target)
        d.ctrl[self._act_ids] = q_target

        return {"pos_err": float(np.linalg.norm(pos_err_vec)),
                "rot_err": float(abs(rot_angle)), "q_target": q_target}


def set_weld_active(model, data, weld_name: str, active: bool) -> bool:
    """[LEGACY/EM] Toggle a weld equality constraint. Superseded by set_suction."""
    import mujoco
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        return False
    try:
        data.eq_active[eq_id] = 1 if active else 0
    except Exception:
        model.eq_active0[eq_id] = 1 if active else 0
    return True


def set_suction(model, data, actuator_name: str, on: bool, level: float = 1.0) -> bool:
    """Set the vacuum `adhesion` actuator ctrl. on=True -> full vacuum."""
    import mujoco
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        return False
    data.ctrl[aid] = float(level) if on else 0.0
    return True


def _cli_smoke(xml_path: str) -> int:
    """Load the KR6 scene, build the bridge, run one action + one servo tick."""
    import mujoco
    print(f"[kr6_ik] loading {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bridge = KR6IKBridge(model, data)
    print(f"[kr6_ik] site_id={bridge._site_id} act_ids={bridge._act_ids.tolist()}")

    action = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    diag = bridge.action_to_ctrl(action)
    print(f"[kr6_ik] ee_pos={np.round(diag['ee_pos'], 4).tolist()} "
          f"q_target={np.round(diag['q_target'], 4).tolist()}")

    tgt = diag["ee_pos"] + np.array([0.0, 0.0, 0.02])
    sdiag = bridge.servo_to_pose(tgt)
    print(f"[kr6_ik] servo pos_err={sdiag['pos_err']:.4f}")

    ok = diag["q_target"].shape == (len(KR6_ARM_JOINTS),)
    print(f"SMOKE_OK={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python kr6_ik_bridge.py <path/to/world.xml>")
        raise SystemExit(2)
    raise SystemExit(_cli_smoke(sys.argv[1]))
