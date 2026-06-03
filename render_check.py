import sys, mujoco
from PIL import Image

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.1"><geom type="box" size="0.08 0.08 0.08" rgba="0.2 0.4 0.9 1"/></body>
    <camera name="cam" pos="0.7 -0.7 0.6" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# Offscreen render — this is what the policy "sees"
r = mujoco.Renderer(m, height=480, width=640)
r.update_scene(d, camera="cam")
img = r.render()
Image.fromarray(img).save("offscreen_test.png")
print("OFFSCREEN_OK shape", img.shape, "-> offscreen_test.png")
r.close()

# Interactive viewer — the window you watch live
if "--viewer" in sys.argv:
    import mujoco.viewer
    print("Opening viewer — close the window to exit.")
    mujoco.viewer.launch(m, d)
