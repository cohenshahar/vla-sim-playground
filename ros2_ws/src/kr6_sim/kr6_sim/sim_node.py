#!/usr/bin/env python3
"""kr6_sim sim_node -- MuJoCo KR6 scene as a ROS2 node.

Publishes:  /joint_states   (sensor_msgs/JointState)  6 arm joints
            /object_states  (geometry_msgs/PoseArray)  [0]=box  [1]=crate
Subscribes: /arm_cmd        (std_msgs/Float64MultiArray) 6 target joint angles
            /vacuum_cmd     (std_msgs/Float64)          adhesion gain (0=off)

Steps physics with gravity compensation on the arm (same as the oracle demo),
opens a viewer, runs ~real time at 50 Hz.

Params:  scene_xml (path), rate (Hz), viewer (bool)
"""
import os
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import PoseArray, Pose

ARM_JOINTS = [f"joint_a{i}" for i in range(1, 7)]
ARM_ACTS   = [f"act_a{i}"   for i in range(1, 7)]
DEFAULT_SCENE = os.path.expanduser("~/Desktop/shahar/vla-sim-playground/scene/world.xml")

# tuning matched to the working oracle
ARM_DAMPING = 40.0; COMP_DAMPING = 15.0; VAC_GAIN = 100.0
BOX_XY = (0.287, 0.035); CRATE_XY = (0.453, -0.162)


class SimNode(Node):
    def __init__(self):
        super().__init__("kr6_sim")
        self.declare_parameter("scene_xml", DEFAULT_SCENE)
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("viewer", True)
        scene = self.get_parameter("scene_xml").value
        self.get_logger().info(f"loading scene: {scene}")
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)

        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        gid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.qadr   = [self.m.jnt_qposadr[jid(j)] for j in ARM_JOINTS]
        self.dofadr = [int(self.m.jnt_dofadr[jid(j)]) for j in ARM_JOINTS]
        self.act    = [aid(a) for a in ARM_ACTS]
        self.vac    = aid("act_vacuum")
        self.box    = bid("metal_box"); self.binb = bid("bin")
        self.boxq   = int(self.m.jnt_qposadr[int(self.m.body_jntadr[self.box])])
        cdof = int(self.m.jnt_dofadr[jid("cup_compliance")])

        # tuning + scene setup (same as the oracle)
        for dd in self.dofadr: self.m.dof_damping[dd] = ARM_DAMPING
        self.m.dof_damping[cdof] = COMP_DAMPING
        self.m.actuator_gainprm[self.vac][0] = VAC_GAIN
        def setg(n, s, p): g = gid(n); self.m.geom_size[g] = s; self.m.geom_pos[g] = p
        setg("bin_floor",   [0.16, 0.16, 0.005], [0, 0, 0.005])
        setg("bin_wall_px", [0.005, 0.16, 0.03], [ 0.16, 0, 0.03])
        setg("bin_wall_nx", [0.005, 0.16, 0.03], [-0.16, 0, 0.03])
        setg("bin_wall_py", [0.16, 0.005, 0.03], [0,  0.16, 0.03])
        setg("bin_wall_ny", [0.16, 0.005, 0.03], [0, -0.16, 0.03])
        self.m.body_pos[self.binb] = [CRATE_XY[0], CRATE_XY[1], 0.85]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.boxq:self.boxq+3] = [BOX_XY[0], BOX_XY[1], 0.90]
        self.d.qpos[self.boxq+3:self.boxq+7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

        self.cmd = np.array(self.d.qpos[self.qadr])   # hold initial pose
        self.vac_cmd = 0.0

        self.js_pub  = self.create_publisher(JointState, "joint_states", 10)
        self.obj_pub = self.create_publisher(PoseArray, "object_states", 10)
        self.create_subscription(Float64MultiArray, "arm_cmd", self.on_arm, 10)
        self.create_subscription(Float64, "vacuum_cmd", self.on_vac, 10)

        self.viewer = None
        if self.get_parameter("viewer").value:
            from mujoco import viewer as _mjv
            self.viewer = _mjv.launch_passive(self.m, self.d)
            self.viewer.cam.lookat[:] = [0.36, -0.06, 0.95]
            self.viewer.cam.distance = 1.8; self.viewer.cam.azimuth = 135; self.viewer.cam.elevation = -22

        rate = float(self.get_parameter("rate").value)
        self.substeps = max(1, int((1.0/rate) / self.m.opt.timestep))
        self.create_timer(1.0/rate, self.tick)
        self.get_logger().info(
            f"sim up: {self.substeps} substeps/tick @ {rate:.0f} Hz | "
            f"pub /joint_states /object_states | sub /arm_cmd /vacuum_cmd")

    def on_arm(self, msg):
        if len(msg.data) == 6:
            self.cmd = np.array(msg.data, float)
        else:
            self.get_logger().warn(f"/arm_cmd needs 6 values, got {len(msg.data)}")

    def on_vac(self, msg):
        self.vac_cmd = float(msg.data)

    def tick(self):
        for _ in range(self.substeps):
            self.d.ctrl[self.act] = self.cmd
            self.d.ctrl[self.vac] = self.vac_cmd
            mujoco.mj_step1(self.m, self.d)
            self.d.qfrc_applied[self.dofadr] = self.d.qfrc_bias[self.dofadr]
            mujoco.mj_step2(self.m, self.d)
        if self.viewer is not None:
            if not self.viewer.is_running():
                rclpy.shutdown(); return
            self.viewer.sync()
        self.publish()

    def publish(self):
        now = self.get_clock().now().to_msg()
        js = JointState(); js.header.stamp = now
        js.name = ARM_JOINTS
        js.position = [float(self.d.qpos[a]) for a in self.qadr]
        js.velocity = [float(self.d.qvel[v]) for v in self.dofadr]
        self.js_pub.publish(js)

        pa = PoseArray(); pa.header.stamp = now; pa.header.frame_id = "world"
        for body in (self.box, self.binb):
            p = Pose()
            xp = self.d.xpos[body]; xq = self.d.xquat[body]
            p.position.x, p.position.y, p.position.z = map(float, xp)
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = map(float, xq)
            pa.poses.append(p)
        self.obj_pub.publish(pa)


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
