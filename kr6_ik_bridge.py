"""
kr6_ik_bridge.py
================
Map an OpenVLA 7-D end-effector action onto the KUKA KR6 R900's six
*position* actuators via damped-least-squares (DLS) inverse kinematics.

WHY THIS FILE EXISTS
--------------------
The generic `mujoco_bridge.openvla_action_to_ctrl(..., env_tag="custom_xml")`
dumps OpenVLA's EE-frame deltas straight into joint-angle targets. For the KR6
that is a category error: the arm just snaps to whatever the raw numbers encode
and jitters. It never reaches toward anything.

OpenVLA emits (BridgeData-v2 convention):
    a[0:3]  delta EE position (x, y, z), expressed in the EE frame, ~[-1, 1]
    a[3:6]  delta EE orientation (roll, pitch, yaw), axis-angle, ~[-1, 1]
    a[6]    gripper (0 = open, 1 = closed; some ckpts use -1/1)

The KR6 scene (VLATraining/sim) exposes:
    - 6 position actuators  act_a1..act_a6  -> target JOINT ANGLES in radians
    - EE reference site      em_contact_site (on em_pad, the EM face centre)
    - EM "grip" = a weld equality constraint `em_weld`, toggled at runtime
      (NOT a ctrl channel)

This bridge closes that gap:
    1. read the current EE pose at `em_contact_site`
    2. turn a[0:3]/a[3:6] into a small desired EE twist in world frame
    3. solve DLS IK on the 6 arm joints -> delta joint angles
    4. write joint-angle targets to data.ctrl[act_a1..a6]
    5. report a gripper intent so the caller can toggle the `em_weld`

STATUS
------
**UNTESTED ON HARDWARE.** Written on the Cowork machine without a GPU or the
KR6 meshes, so it has not been run against the real MuJoCo model yet. It is a
draft to validate on the strong PC. Do not treat its numbers as known-good
until a smoke run confirms shapes and motion.

Tunables you will likely adjust on first run: `pos_step`, `rot_step`,
`damping`, `max_joint_step`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# Default name wiring for the KR6 scene (VLATraining/sim/scene).
KR6_EE_SITE = "em_contact_site"
KR6_ARM_JOINTS = ("joint_a1", "joint_a2", "joint_a3",
                  "joint_a4", "joint_a5", "joint_a6")
KR6_ARM_ACTUATORS = ("act_a1", "act_a2", "act_a3",
                     "act_a4", "act_a5", "act_a6")
KR6_EM_WELD = "em_weld"


@dataclass
class KR6IKBridge:
    """DLS IK bridge from OpenVLA 7-D action to KR6 joint-angle targets.

    Args:
        model, data    : mujoco.MjModel / MjData for the loaded KR6 scene.
        ee_site        : name of the EE reference site.
        arm_joints     : ordered joint names (must be 1-DoF hinges).
        arm_actuators  : ordered position-actuator names, aligned to arm_joints.
        pos_step       : metres of EE translation per unit of a[0:3].
        rot_step       : radians of EE rotation per unit of a[3:6].
        damping        : DLS lambda (larger = smoother + more stable, less exact).
        max_joint_step : per-call clamp on |delta q| (rad) to avoid jumps.
        use_orientation: include a[3:6] in the IK task (False = position only,
                         more stable for a first bring-up).
        gripper_threshold: a[6] above this => "closed" intent.
    """

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

    # resolved at __post_init__
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
        self._jnt_range = np.asarray(jnt_range, dtype=float)  # (6, 2)

        act_ids = []
        for aname in self.arm_actuators:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"actuator {aname!r} not found in model.")
            act_ids.append(int(aid))
        self._act_ids = np.asarray(act_ids, dtype=int)

        n = len(self.arm_joints)
        if not (len(self.arm_actuators) == n):
            raise ValueError("arm_joints and arm_actuators must be the same length.")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def action_to_ctrl(self, action: np.ndarray) -> dict:
        """Translate one OpenVLA action into KR6 joint-angle targets and apply.

        Side effect: writes data.ctrl[act_a1..a6] = new joint targets.

        Returns a diagnostics dict:
            {
              "q_cur": (6,), "q_target": (6,), "dq": (6,),
              "ee_pos": (3,), "task_err": (6,),
              "gripper_closed": bool, "gripper_raw": float,
            }
        """
        import mujoco

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != 7:
            raise ValueError(f"expected 7-D action, got shape={a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite action: {a.tolist()}")

        m, d = self.model, self.data

        # --- current EE pose ------------------------------------------
        ee_pos = np.array(d.site_xpos[self._site_id], dtype=float)        # (3,)
        ee_mat = np.array(d.site_xmat[self._site_id], dtype=float).reshape(3, 3)

        # --- desired EE twist (world frame) ---------------------------
        # OpenVLA deltas are in the EE frame; rotate into world by ee_mat.
        dpos_ee = self.pos_step * a[0:3]
        dpos_world = ee_mat @ dpos_ee
        if self.use_orientation:
            drot_ee = self.rot_step * a[3:6]
            drot_world = ee_mat @ drot_ee
            task_err = np.concatenate([dpos_world, drot_world])           # (6,)
        else:
            task_err = dpos_world                                         # (3,)

        # --- Jacobian at the EE site (world frame, all dofs) ----------
        jacp = np.zeros((3, m.nv), dtype=float)
        jacr = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacSite(m, d, jacp, jacr, self._site_id)
        Jp = jacp[:, self._dof_adr]                                       # (3, 6)
        if self.use_orientation:
            Jr = jacr[:, self._dof_adr]                                   # (3, 6)
            J = np.vstack([Jp, Jr])                                       # (6, 6)
        else:
            J = Jp                                                        # (3, 6)

        # --- damped least squares: dq = J^T (J J^T + l^2 I)^-1 e -------
        lam2 = self.damping ** 2
        JJt = J @ J.T
        A = JJt + lam2 * np.eye(JJt.shape[0])
        dq = J.T @ np.linalg.solve(A, task_err)                          # (6,)

        # clamp per-joint step to avoid violent jumps
        np.clip(dq, -self.max_joint_step, self.max_joint_step, out=dq)

        # --- new joint targets ----------------------------------------
        q_cur = d.qpos[self._qpos_adr].copy()                           # (6,)
        q_target = q_cur + dq
        lo, hi = self._jnt_range[:, 0], self._jnt_range[:, 1]
        # only clip where a real range is declared (lo < hi)
        has_range = hi > lo
        q_target = np.where(has_range, np.clip(q_target, lo, hi), q_target)

        # --- apply to position actuators ------------------------------
        d.ctrl[self._act_ids] = q_target

        gripper_raw = float(a[6])
        gripper_closed = gripper_raw > self.gripper_threshold

        return {
            "q_cur": q_cur,
            "q_target": q_target,
            "dq": dq,
            "ee_pos": ee_pos,
            "task_err": task_err,
            "gripper_closed": gripper_closed,
            "gripper_raw": gripper_raw,
        }


# ----------------------------------------------------------------------
# Minimal gripper helper (naive — no proximity check)
# ----------------------------------------------------------------------
def set_weld_active(model, data, weld_name: str, active: bool) -> bool:
    """Toggle a weld equality constraint by name. Returns True on success.

    NOTE: this is the *naive* version — it does not run the proximity /
    orientation checks that em_controller.py uses in the thesis sim. For a
    faithful grasp, port em_controller's activate/release logic instead.
    """
    import mujoco

    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        return False
    # data.eq_active is the runtime flag (MuJoCo 3.x).
    try:
        data.eq_active[eq_id] = 1 if active else 0
    except Exception:
        # Fallback for older bindings that only expose model.eq_active0.
        model.eq_active0[eq_id] = 1 if active else 0
    return True


# ----------------------------------------------------------------------
# CLI smoke (runs only if a model XML is given AND it loads)
# ----------------------------------------------------------------------
def _cli_smoke(xml_path: str) -> int:
    """Load the KR6 scene, build the bridge, run ONE step on a zero action.

    This validates name wiring + tensor shapes. It does NOT validate motion
    quality. Requires the KR6 meshes to resolve (see arm.xml compiler meshdir).
    """
    import mujoco

    print(f"[kr6_ik] loading {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bridge = KR6IKBridge(model, data)
    print(f"[kr6_ik] site_id={bridge._site_id} act_ids={bridge._act_ids.tolist()}")

    # small forward+up nudge, no rotation, gripper open
    action = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    diag = bridge.action_to_ctrl(action)
    print(f"[kr6_ik] ee_pos    = {np.round(diag['ee_pos'], 4).tolist()}")
    print(f"[kr6_ik] dq        = {np.round(diag['dq'], 4).tolist()}")
    print(f"[kr6_ik] q_target  = {np.round(diag['q_target'], 4).tolist()}")
    ok = diag["q_target"].shape == (len(KR6_ARM_JOINTS),)
    print(f"SMOKE_OK={int(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python kr6_ik_bridge.py <path/to/world.xml>")
        raise SystemExit(2)
    raise SystemExit(_cli_smoke(sys.argv[1]))
