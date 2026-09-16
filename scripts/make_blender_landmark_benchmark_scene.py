import argparse
import csv
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Create a Blender scene for manual-vs-predicted landmark matches.")
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--source-geometry", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--target-geometry", required=True)
    parser.add_argument("--benchmark-csv", required=True, help="CSV from evaluate_landmark_benchmark.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gap", type=float, default=2.2)
    parser.add_argument("--marker-radius", type=float, default=0.035)
    parser.add_argument("--line-radius", type=float, default=0.006)
    parser.add_argument("--max-landmarks", type=int, default=50)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_geometry(path):
    path = str(Path(path).resolve())
    if path.lower().endswith(".ply"):
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif path.lower().endswith(".obj"):
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise ValueError(f"Unsupported geometry extension: {path}")
    return bpy.context.object


def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.72
        bsdf.inputs["Alpha"].default_value = color[3]
    material.blend_method = "BLEND"
    return material


def assign_material(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)


def object_width(obj):
    xs = [corner[0] for corner in obj.bound_box]
    return max(xs) - min(xs)


def normalize_and_place(source, target, gap):
    max_dim = max(
        source.dimensions.x,
        source.dimensions.y,
        source.dimensions.z,
        target.dimensions.x,
        target.dimensions.y,
        target.dimensions.z,
    )
    scale = 3.0 / max(max_dim, 1e-6)
    for obj in (source, target):
        obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    for obj in (source, target):
        center = sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8.0
        obj.location.x -= center.x
        obj.location.y -= center.y
        obj.location.z -= center.z
    bpy.context.view_layer.update()

    source.location.x -= gap / 2.0
    target.location.x += gap / 2.0
    bpy.context.view_layer.update()
    return scale


def local_to_world(obj, x, y, z):
    return obj.matrix_world @ Vector((float(x), float(y), float(z)))


def add_marker(location, radius, material, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_line(start, end, radius, material, name):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (start.x, start.y, start.z, 1.0)
    spline.points[1].co = (end.x, end.y, end.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def add_label(text, location, size, material):
    bpy.ops.object.text_add(location=location, rotation=(math.radians(75), 0, 0))
    label = bpy.context.object
    label.name = f"label_{text}"
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.data.size = size
    label.data.materials.append(material)
    return label


def read_rows(path, max_landmarks):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("target_error_bbox", 0.0)), reverse=True)
    return rows[:max_landmarks]


def add_camera_and_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -4.5, 5.0))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 750
    light.data.size = 5

    bpy.ops.object.camera_add(location=(0, -7.3, 3.1), rotation=(math.radians(63), 0, 0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 6.4
    bpy.context.scene.camera = camera


def set_view_settings():
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def main():
    args = parse_args()
    clear_scene()

    source = import_geometry(args.source_geometry)
    source.name = args.source_label
    target = import_geometry(args.target_geometry)
    target.name = args.target_label

    assign_material(source, make_material("source_soft_gray", (0.72, 0.74, 0.76, 0.34)))
    assign_material(target, make_material("target_soft_gray", (0.72, 0.74, 0.76, 0.34)))
    normalize_and_place(source, target, args.gap)

    source_mat = make_material("manual_source_blue", (0.10, 0.28, 0.95, 1.0))
    gt_mat = make_material("manual_target_green", (0.10, 0.66, 0.30, 1.0))
    pred_mat = make_material("predicted_target_red", (0.92, 0.14, 0.10, 1.0))
    match_line_mat = make_material("source_to_prediction_line", (0.08, 0.42, 0.46, 0.55))
    error_line_mat = make_material("prediction_error_line", (0.95, 0.12, 0.08, 0.95))
    text_mat = make_material("label_dark", (0.02, 0.02, 0.02, 1.0))

    rows = read_rows(args.benchmark_csv, args.max_landmarks)
    for i, row in enumerate(rows):
        label = row["label"]
        source_point = local_to_world(source, row["source_x"], row["source_y"], row["source_z"])
        target_gt = local_to_world(target, row["target_gt_x"], row["target_gt_y"], row["target_gt_z"])
        target_pred = local_to_world(target, row["pred_target_x"], row["pred_target_y"], row["pred_target_z"])

        add_marker(source_point, args.marker_radius, source_mat, f"source_manual_{i:03d}_{label}")
        add_marker(target_gt, args.marker_radius, gt_mat, f"target_manual_{i:03d}_{label}")
        add_marker(target_pred, args.marker_radius * 0.78, pred_mat, f"target_predicted_{i:03d}_{label}")
        add_line(source_point, target_pred, args.line_radius, match_line_mat, f"source_to_prediction_{i:03d}_{label}")
        add_line(target_gt, target_pred, args.line_radius * 1.25, error_line_mat, f"target_error_{i:03d}_{label}")
        add_label(label, target_gt + Vector((0, -0.06, args.marker_radius * 2.0)), args.marker_radius * 3.0, text_mat)

    add_label(args.source_label, (source.location.x, -2.25, -1.85), 0.24, text_mat)
    add_label(args.target_label, (target.location.x, -2.25, -1.85), 0.24, text_mat)
    add_camera_and_lights()
    set_view_settings()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    print(f"Saved {output}")
    print(f"Landmarks in scene: {len(rows)}")


if __name__ == "__main__":
    main()
