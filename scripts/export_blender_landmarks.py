"""Write the LM_<name> marker positions of the open Blender file to a landmark CSV.

    blender labels.blend --background --python scripts/export_blender_landmarks.py -- \
        --output landmarks/brownCricket.csv

By default (--coordinate-space blender-obj) x,y,z are converted back from
Blender's Z-up world frame to the Y-up frame of an OBJ file, which is the frame
of the mesh the descriptor was computed on: an OBJ imported into Blender, or
one exported from Blender after rotating it. The CSV also keeps the raw world
and object-local coordinates.
"""

import argparse
import csv
from pathlib import Path
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export Blender landmark marker positions to CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--mesh-object", help="Geometry object name. If omitted, infer the only non-landmark mesh.")
    parser.add_argument("--marker-prefix", default="LM_")
    parser.add_argument(
        "--coordinate-space",
        choices=("blender-obj", "world", "local"),
        default="blender-obj",
        help="Frame of the x,y,z columns: OBJ file axes (default), Blender world, or the object's local frame.",
    )
    return parser.parse_args(argv)


def infer_mesh_object(marker_prefix):
    candidates = []
    for obj in bpy.context.scene.objects:
        if obj.name.startswith(marker_prefix):
            continue
        if obj.type == "MESH" and hasattr(obj.data, "vertices") and len(obj.data.vertices) > 0:
            candidates.append(obj)
    if len(candidates) != 1:
        names = ", ".join(obj.name for obj in candidates)
        raise ValueError(
            "Could not infer geometry object. "
            f"Use --mesh-object. Candidate mesh objects: {names}"
        )
    return candidates[0]


def main():
    args = parse_args()
    if args.mesh_object:
        mesh_object = bpy.data.objects.get(args.mesh_object)
        if mesh_object is None:
            raise ValueError(f"Could not find mesh object named {args.mesh_object!r}.")
    else:
        mesh_object = infer_mesh_object(args.marker_prefix)

    inverse = mesh_object.matrix_world.inverted()
    rows = []
    for obj in bpy.context.scene.objects:
        if not obj.name.startswith(args.marker_prefix):
            continue
        label = obj.name[len(args.marker_prefix) :]
        world = obj.matrix_world.translation
        local = inverse @ world

        if args.coordinate_space == "local":
            xyz = local
        elif args.coordinate_space == "world":
            xyz = world
        else:
            xyz = (world.x, world.z, -world.y)

        rows.append(
            {
                "label": label,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "coordinate_space": args.coordinate_space,
                "local_x": float(local.x),
                "local_y": float(local.y),
                "local_z": float(local.z),
                "world_x": float(world.x),
                "world_y": float(world.y),
                "world_z": float(world.z),
                "mesh_object": mesh_object.name,
                "marker_object": obj.name,
            }
        )

    if not rows:
        raise ValueError(f"No landmark objects found with prefix {args.marker_prefix!r}.")
    rows.sort(key=lambda row: row["label"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {output}")
    print(f"Landmarks: {len(rows)}")
    print(f"Geometry object: {mesh_object.name}")


if __name__ == "__main__":
    main()
