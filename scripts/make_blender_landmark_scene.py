import argparse
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

    parser = argparse.ArgumentParser(description="Create a Blender scene for manually placing landmarks.")
    parser.add_argument("--geometry", required=True, help="Mesh or point-cloud OBJ/PLY to label.")
    parser.add_argument("--label", required=True, help="Object/specimen label, for example bcrick.")
    parser.add_argument("--landmark", nargs="+", required=True, help="Landmark names to create as movable markers.")
    parser.add_argument("--output", required=True, help="Output .blend path.")
    parser.add_argument("--marker-prefix", default="LM_")
    parser.add_argument("--marker-radius", type=float, default=None)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_geometry(path):
    path = str(Path(path).resolve())
    suffix = path.lower()
    if suffix.endswith(".ply"):
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif suffix.endswith(".obj"):
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
        bsdf.inputs["Roughness"].default_value = 0.74
        bsdf.inputs["Alpha"].default_value = color[3]
    material.blend_method = "BLEND"
    return material


def assign_material(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    maxs = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mins, maxs


def bbox_diagonal(mins, maxs):
    return max((maxs - mins).length, 1e-6)


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


def add_marker(name, location, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=location)
    marker = bpy.context.object
    marker.name = name
    marker.data.name = f"{name}_mesh"
    marker.data.materials.append(material)
    return marker


def add_text(name, text, location, size, material):
    bpy.ops.object.text_add(location=location, rotation=(math.radians(70), 0, 0))
    label = bpy.context.object
    label.name = name
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.data.size = size
    label.data.materials.append(material)
    return label


def add_camera_and_light(mins, maxs):
    center = (mins + maxs) * 0.5
    diagonal = bbox_diagonal(mins, maxs)
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - diagonal * 1.2, center.z + diagonal * 0.9))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 650
    light.data.size = diagonal * 0.8

    bpy.ops.object.camera_add(
        location=(center.x, center.y - diagonal * 1.8, center.z + diagonal * 0.75),
        rotation=(math.radians(65), 0, 0),
    )
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = diagonal * 1.15
    bpy.context.scene.camera = camera


def main():
    args = parse_args()
    clear_scene()

    geometry = import_geometry(args.geometry)
    geometry.name = args.label
    assign_material(geometry, make_material("geometry_soft_gray", (0.72, 0.74, 0.76, 0.42)))
    mins, maxs = world_bbox(geometry)
    diagonal = bbox_diagonal(mins, maxs)
    radius = args.marker_radius if args.marker_radius is not None else diagonal * 0.012
    label_material = make_material("label_dark", (0.02, 0.02, 0.02, 1.0))

    start = Vector((mins.x - diagonal * 0.08, mins.y, maxs.z + diagonal * 0.05))
    spacing = diagonal * 0.045
    for i, landmark in enumerate(args.landmark):
        material = make_material(f"{args.marker_prefix}{landmark}_mat", hsv_color(i, len(args.landmark)))
        location = start + Vector((0, 0, -i * spacing))
        add_marker(f"{args.marker_prefix}{landmark}", location, radius, material)
        add_text(f"TXT_{landmark}", landmark, location + Vector((radius * 2.2, 0, 0)), radius * 3.8, label_material)

    add_camera_and_light(mins, maxs)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    print(f"Saved {output}")
    print(f"Move objects named {args.marker_prefix}<label> onto the anatomy, then save and export landmarks.")


if __name__ == "__main__":
    main()
