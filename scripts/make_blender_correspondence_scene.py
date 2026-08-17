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

    parser = argparse.ArgumentParser(description="Create a Blender scene for feature correspondences.")
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--target-mesh", required=True)
    parser.add_argument("--matches", required=True, help="CSV from compute_feature_correspondences.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-matches", type=int, default=80)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--only-mutual", action="store_true")
    parser.add_argument("--gap", type=float, default=2.2)
    parser.add_argument("--marker-radius", type=float, default=0.035)
    parser.add_argument("--line-radius", type=float, default=0.006)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(path):
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
        raise ValueError(f"Unsupported mesh extension: {path}")
    return bpy.context.object


def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.72
    return material


def make_mesh_material(name, color):
    material = make_material(name, color)
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Alpha"].default_value = color[3]
    material.blend_method = "BLEND"
    material.use_screen_refraction = True
    return material


def assign_material(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)


def normalize_and_place(source, target, gap):
    max_dim = max(source.dimensions.x, source.dimensions.y, source.dimensions.z, target.dimensions.x, target.dimensions.y, target.dimensions.z)
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


def transformed_vertex(obj, index):
    return obj.matrix_world @ obj.data.vertices[index].co


def hsv_color(i, n):
    hue = i / max(n, 1)
    c = 1.0
    x = 1.0 - abs((hue * 6.0) % 2.0 - 1.0)
    if hue < 1 / 6:
        rgb = (c, x, 0)
    elif hue < 2 / 6:
        rgb = (x, c, 0)
    elif hue < 3 / 6:
        rgb = (0, c, x)
    elif hue < 4 / 6:
        rgb = (0, x, c)
    elif hue < 5 / 6:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)
    return (*rgb, 1.0)


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


def read_matches(path, max_matches, min_score, only_mutual):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            score = float(row["cosine_score"])
            mutual = row.get("mutual", "False").lower() == "true"
            if score < min_score:
                continue
            if only_mutual and not mutual:
                continue
            rows.append(row)
    rows.sort(key=lambda row: float(row["cosine_score"]), reverse=True)
    return rows[:max_matches]


def add_label(text, location):
    bpy.ops.object.text_add(location=location, rotation=(math.radians(75), 0, 0))
    label = bpy.context.object
    label.name = f"label_{text}"
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.data.size = 0.24
    label.data.materials.append(make_material(f"{label.name}_mat", (0.02, 0.02, 0.02, 1.0)))


def add_camera_and_lights():
    bpy.ops.object.light_add(type="AREA", location=(0, -4.5, 5.0))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 700
    light.data.size = 5

    bpy.ops.object.camera_add(location=(0, -7.2, 3.1), rotation=(math.radians(63), 0, 0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 6.3
    camera.data.lens = 30
    bpy.context.scene.camera = camera


def set_view_settings():
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def main():
    args = parse_args()
    clear_scene()

    source = import_mesh(args.source_mesh)
    source.name = args.source_label
    target = import_mesh(args.target_mesh)
    target.name = args.target_label

    assign_material(source, make_mesh_material("source_soft_gray", (0.72, 0.74, 0.76, 0.34)))
    assign_material(target, make_mesh_material("target_soft_gray", (0.72, 0.74, 0.76, 0.34)))
    scale = normalize_and_place(source, target, args.gap)

    matches = read_matches(args.matches, args.max_matches, args.min_score, args.only_mutual)
    for i, row in enumerate(matches):
        material = make_material(f"match_{i:03d}_mat", hsv_color(i, len(matches)))
        source_point = transformed_vertex(source, int(row["source_index"]))
        target_point = transformed_vertex(target, int(row["target_index"]))
        add_marker(source_point, args.marker_radius, material, f"source_match_{i:03d}")
        add_marker(target_point, args.marker_radius, material, f"target_match_{i:03d}")
        add_line(source_point, target_point, args.line_radius, material, f"line_match_{i:03d}")

    add_label(args.source_label, (source.location.x, -2.25, -1.85))
    add_label(args.target_label, (target.location.x, -2.25, -1.85))
    add_camera_and_lights()
    set_view_settings()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    print(f"Saved {output}")
    print(f"Matches in scene: {len(matches)}")
    print(f"Applied scale: {scale:.6f}")


if __name__ == "__main__":
    main()
