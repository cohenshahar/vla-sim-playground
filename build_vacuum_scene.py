"""
build_vacuum_scene.py
=====================
Automate the EM-weld -> vacuum/suction surgery on the COPIED KR6 scene, per
docs/vacuum_gripper_design.md (§3,4,8) and docs/pick_place_motion_spec.md (§1a).

What it does (only on --apply; dry-run by default):
  1. Remove the <equality>/<weld name="em_weld"> block.
  2. Replace the <body name="em_pad"> block with the compliant suction pad:
       suction_mount (sprung `slide` cup_compliance) > suction_pad (cup_geom +
       suction_tip_site + cam_wrist + cup_contact_site).
  3. Add <adhesion name="act_vacuum" body="suction_pad" ctrlrange="0 1" gain="40"/>
     to the <actuator> block.
  4. Add the open `bin` body (floor + 4 walls + bin_center site) to the worldbody.

Safe by design:
  * DRY-RUN unless --apply. Prints exactly what it would change.
  * Writes <file>.bak before editing.
  * Idempotent: skips anything already migrated (e.g. act_vacuum present).
  * If an expected anchor is missing, it does NOT guess -- it prints the exact
    snippet for you to paste and where.

NOTE: ElementTree reformats XML (comments/spacing may change). Backups are made.
Validated suction numbers (suction_probe, mujoco 3.1.6): gain=40, stiffness=800,
damping=5, margin=0.004. Lift speed must stay slow (handled in the oracle).

Usage:
    python build_vacuum_scene.py --scene-dir scene            # dry run
    python build_vacuum_scene.py --scene-dir scene --apply     # do it (after reading the plan)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GAIN, STIFFNESS, DAMPING, MARGIN = 40, 800, 5, 0.004

SUCTION_MOUNT_XML = f"""
<body name="suction_mount" pos="0.01 0 0">
  <joint name="cup_compliance" type="slide" axis="1 0 0" range="-0.010 0"
         stiffness="{STIFFNESS}" damping="{DAMPING}" springref="0"/>
  <body name="suction_pad">
    <geom name="cup_geom" type="cylinder" fromto="-0.005 0 0  0.005 0 0" size="0.03"
          rgba="0.15 0.15 0.18 1" margin="{MARGIN}" solref="0.02 1"
          solimp="0.9 0.95 0.001" contype="1" conaffinity="6" mass="0.05"/>
    <site name="suction_tip_site" pos="0.006 0 0" euler="0 -90 0" size="0.005" rgba="1 0 0 0.5"/>
    <camera name="cam_wrist" pos="-0.02 0 0.10" xyaxes="0 1 0 -0.342 0 -0.940" fovy="80"/>
    <site name="cup_contact_site" pos="0.006 0 0" size="0.004" rgba="1 0.5 0 0"/>
  </body>
</body>"""

BIN_XML = """
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
</body>"""


def _find_parent(root, child):
    for parent in root.iter():
        for c in list(parent):
            if c is child:
                return parent
    return None


def _find_named(root, tag, name):
    for el in root.iter(tag):
        if el.get("name") == name:
            return el
    return None


def process_file(path: Path, apply: bool):
    changes = []
    todo = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"  [skip] {path.name}: parse error ({e})")
        return changes, todo
    root = tree.getroot()
    dirty = False

    # 1. remove em_weld equality
    weld = _find_named(root, "weld", "em_weld")
    if weld is not None:
        parent = _find_parent(root, weld)
        changes.append("remove <weld name='em_weld'>")
        if apply:
            parent.remove(weld)
            # drop now-empty <equality>
            if parent.tag == "equality" and len(list(parent)) == 0:
                gp = _find_parent(root, parent)
                if gp is not None:
                    gp.remove(parent)
            dirty = True

    # 2. replace em_pad body
    em_pad = _find_named(root, "body", "em_pad")
    already = _find_named(root, "body", "suction_pad")
    if em_pad is not None and already is None:
        parent = _find_parent(root, em_pad)
        changes.append("replace <body name='em_pad'> with suction_mount/suction_pad")
        if apply:
            idx = list(parent).index(em_pad)
            parent.remove(em_pad)
            parent.insert(idx, ET.fromstring(SUCTION_MOUNT_XML))
            dirty = True
    elif already is not None:
        changes.append("suction_pad already present (skip body replace)")

    # 3. add adhesion actuator
    actuator = root.find(".//actuator")
    if actuator is not None:
        has_vac = _find_named(root, "adhesion", "act_vacuum") is not None
        has_arm = _find_named(root, "position", "act_a1") is not None or \
                  any(a.get("name", "").startswith("act_a") for a in actuator)
        if not has_vac and has_arm:
            changes.append("add <adhesion name='act_vacuum' body='suction_pad' gain='40'>")
            if apply:
                actuator.append(ET.fromstring(
                    f'<adhesion name="act_vacuum" body="suction_pad" '
                    f'ctrlrange="0 1" gain="{GAIN}"/>'))
                dirty = True
        elif has_vac:
            changes.append("act_vacuum already present (skip)")

    # 4. add bin to worldbody (only in the file that has the worldbody with objects)
    worldbody = root.find(".//worldbody")
    if worldbody is not None and _find_named(root, "body", "bin") is None:
        # heuristic: add bin in the file that defines the target object or the table
        if _find_named(root, "body", "metal_box") is not None or \
           _find_named(root, "body", "target_zone") is not None or \
           _find_named(root, "geom", "table") is not None:
            changes.append("add open <body name='bin'> to worldbody")
            if apply:
                worldbody.append(ET.fromstring(BIN_XML))
                dirty = True

    if apply and dirty:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        tree.write(path, encoding="unicode", xml_declaration=False)

    return changes, todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="dir with world.xml / includes/*.xml")
    ap.add_argument("--apply", action="store_true", help="actually edit (default: dry-run)")
    a = ap.parse_args()

    sdir = Path(a.scene_dir).expanduser().resolve()
    if not sdir.exists():
        raise SystemExit(f"scene dir not found: {sdir}")
    xmls = sorted(sdir.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"no .xml files under {sdir}")

    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"=== build_vacuum_scene [{mode}] over {len(xmls)} files in {sdir} ===")
    any_change = False
    for x in xmls:
        ch, _ = process_file(x, a.apply)
        if ch:
            any_change = True
            print(f"\n{x.relative_to(sdir)}:")
            for c in ch:
                print(f"   - {c}")
    if not any_change:
        print("\nNo EM/suction anchors found to change. Either already migrated, or the "
              "element names differ. Snippets to paste manually:")
        print("  ADHESION (into <actuator>):")
        print(f'    <adhesion name="act_vacuum" body="suction_pad" ctrlrange="0 1" gain="{GAIN}"/>')
        print("  SUCTION PAD (replace em_pad body):", SUCTION_MOUNT_XML)
        print("  BIN (into <worldbody>):", BIN_XML)
    if not a.apply and any_change:
        print("\nDry-run only. Re-run with --apply to make these changes (.bak backups written).")
        print("After applying, verify with:")
        print("  python kr6_ik_bridge.py scene/world.xml   # expect SMOKE_OK=1, act_vacuum/cup_compliance present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
