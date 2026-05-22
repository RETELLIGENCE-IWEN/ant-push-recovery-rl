"""
M0: Load MuJoCo Menagerie Unitree Go1 model and dump its structural properties.

Goal: catalogue what we'll have to handle differently from Ant-v5 before
running any policy. Saves the inventory to reports/go1_model_inspection.json
and produces a passive-drop video (no control) to videos/go1_passive_drop.mp4
to see how the freely-falling model behaves under gravity.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import imageio
import mujoco
import numpy as np

MODEL_SCENE = Path("/workspace/external/mujoco_menagerie/unitree_go1/scene.xml")
OUT_JSON = Path("reports/go1_model_inspection.json")
OUT_VIDEO = Path("videos/go1_passive_drop.mp4")


def name_list(model, count, getter):
    names = []
    for i in range(count):
        name = getter(i).name
        names.append(name if name else f"<unnamed-{i}>")
    return names


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_SCENE))
    data = mujoco.MjData(model)

    body_masses = []
    for i in range(model.nbody):
        body_masses.append(
            {"id": i, "name": model.body(i).name or f"<body-{i}>", "mass": float(model.body_mass[i])}
        )

    joints = []
    for i in range(model.njnt):
        j = model.jnt(i)
        joints.append({
            "id": i,
            "name": j.name,
            "type": int(model.jnt_type[i]),  # 0=free, 1=ball, 2=slide, 3=hinge
            "qposadr": int(model.jnt_qposadr[i]),
            "dofadr": int(model.jnt_dofadr[i]),
            "range_min": float(model.jnt_range[i][0]),
            "range_max": float(model.jnt_range[i][1]),
            "limited": bool(model.jnt_limited[i]),
        })

    actuators = []
    for i in range(model.nu):
        a = model.actuator(i)
        actuators.append({
            "id": i,
            "name": a.name,
            "ctrlrange_min": float(model.actuator_ctrlrange[i][0]),
            "ctrlrange_max": float(model.actuator_ctrlrange[i][1]),
            "forcerange_min": float(model.actuator_forcerange[i][0]),
            "forcerange_max": float(model.actuator_forcerange[i][1]),
            "gainprm": [float(x) for x in model.actuator_gainprm[i][:3]],
            "biasprm": [float(x) for x in model.actuator_biasprm[i][:3]],
            "trntype": int(model.actuator_trntype[i]),
        })

    sites = []
    for i in range(model.nsite):
        s = model.site(i)
        sites.append({
            "id": i,
            "name": s.name,
            "bodyid": int(model.site_bodyid[i]),
            "body_name": model.body(int(model.site_bodyid[i])).name,
        })

    # Find foot bodies / sites by name heuristic
    foot_keywords = ["foot", "toe"]
    foot_bodies = [b for b in body_masses if any(k in b["name"].lower() for k in foot_keywords)]
    foot_sites = [s for s in sites if any(k in s["name"].lower() for k in foot_keywords)]

    inspection = {
        "model_path": str(MODEL_SCENE),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "timestep": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "total_mass": float(sum(b["mass"] for b in body_masses)),
        "trunk_mass": next(
            (b["mass"] for b in body_masses if b["name"] == "trunk"), None
        ),
        "init_qpos_first_7": [float(x) for x in model.qpos0[:7]],
        "init_qpos_joints": [float(x) for x in model.qpos0[7:]],
        "body_masses": body_masses,
        "joints": joints,
        "actuators": actuators,
        "sites": sites,
        "foot_bodies": foot_bodies,
        "foot_sites": foot_sites,
    }

    with OUT_JSON.open("w") as f:
        json.dump(inspection, f, indent=2)
    print(f"[M0] saved inspection JSON: {OUT_JSON}")

    # Summary stdout
    print("\n=== Go1 model summary ===")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")
    print(f"  timestep={model.opt.timestep}s  ->  control freq if every step = {1/model.opt.timestep:.0f}Hz")
    print(f"  total mass={inspection['total_mass']:.3f} kg, trunk={inspection['trunk_mass']:.3f} kg")
    print(f"  free joint qpos: pos(3) + quat(4) at indices 0..6")
    print(f"  joint qpos starting at index 7, count = {model.nq - 7}")
    print(f"  actuators ({model.nu}):")
    for a in actuators:
        print(f"    {a['name']:24s} ctrl[{a['ctrlrange_min']:+.3f}, {a['ctrlrange_max']:+.3f}]  "
              f"force[{a['forcerange_min']:+.2f}, {a['forcerange_max']:+.2f}]  "
              f"trntype={a['trntype']}  gainprm[0]={a['gainprm'][0]}")
    print(f"  joints ({model.njnt}):")
    for j in joints:
        types = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        print(f"    {j['name']:24s} type={types[j['type']]:5s} "
              f"range=[{j['range_min']:+.3f},{j['range_max']:+.3f}]  "
              f"qposadr={j['qposadr']}  dofadr={j['dofadr']}")
    print(f"  candidate foot bodies: {[b['name'] for b in foot_bodies]}")
    print(f"  candidate foot sites:  {[s['name'] for s in foot_sites]}")

    # Passive drop video: no control applied, see how the model settles under gravity.
    print(f"\n[M0] recording passive drop video -> {OUT_VIDEO}")
    renderer = mujoco.Renderer(model, height=480, width=640)
    duration_s = 3.0
    fps = 30
    sim_steps_per_frame = max(int((1.0 / fps) / model.opt.timestep), 1)
    total_frames = int(duration_s * fps)

    mujoco.mj_resetData(model, data)
    with imageio.get_writer(str(OUT_VIDEO), fps=fps) as writer:
        for f_idx in range(total_frames):
            for _ in range(sim_steps_per_frame):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="tracking")
            frame = renderer.render()
            writer.append_data(frame)
    renderer.close()
    print(f"[M0] video saved ({total_frames} frames @ {fps}fps = {duration_s}s)")
    print(f"[M0] post-drop z = {float(data.qpos[2]):.3f}, "
          f"roll/pitch/yaw from quat = (not computed here)")


if __name__ == "__main__":
    main()
