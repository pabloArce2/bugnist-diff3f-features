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

    parser = argparse.ArgumentParser(description="Create a Blender scene comparing colored feature PLY meshes.")
    parser.add_argument("--item", nargs=2, action="append", metavar=("LABEL", "PLY"), required=True)
    parser.add_argument("--output", required=True, help="Output .blend path.")
    parser.add_argument("--gap", type=float, default=1.4)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_ply(path):
    path = str(Path(path).resolve())
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        bpy.ops.import_mesh.ply(filepath=path)
    return bpy.context.object


def apply_vertex_color_material(obj):
    material = bpy.data.materials.new(f"{obj.name}_vertex_colors")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        obj.data.materials.append(material)
        return

    color_name = None
    if hasattr(obj.data, "color_attributes") and obj.data.color_attributes:
        color_name = obj.data.color_attributes[0].name
    elif hasattr(obj.data, "vertex_colors") and obj.data.vertex_colors:
        color_name = obj.data.vertex_colors[0].name

    if color_name:
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.attribute_name = color_name
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(material)


def object_width(obj):
    xs = [corner[0] for corner in obj.bound_box]
    return max(xs) - min(xs)


def normalize_and_place(objects, gap):
    max_dim = 0.0
    for obj in objects:
        dims = obj.dimensions
        max_dim = max(max_dim, dims.x, dims.y, dims.z)
    scale = 3.0 / max(max_dim, 1e-6)

    widths = []
    for obj in objects:
        obj.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        widths.append(object_width(obj))

    cursor = 0.0
    for obj, width in zip(objects, widths):
        center = sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8.0
        obj.location.x += cursor - center.x
        obj.location.y -= center.y
        obj.location.z -= center.z
        cursor += width + gap

    total_width = cursor - gap if objects else 0.0
    for obj in objects:
        obj.location.x -= total_width / 2.0
    bpy.context.view_layer.update()
    return total_width


def add_label(text, obj):
    bpy.ops.object.text_add(location=(obj.location.x, -2.2, -1.9), rotation=(math.radians(75), 0, 0))
    label = bpy.context.object
    label.name = f"label_{text}"
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.data.size = 0.22
    mat = bpy.data.materials.new(f"{label.name}_mat")
    mat.diffuse_color = (0.02, 0.02, 0.02, 1.0)
    label.data.materials.append(mat)


def add_camera_and_lights(total_width):
    bpy.ops.object.light_add(type="AREA", location=(0, -4.0, 5.0))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 600
    light.data.size = 5

    bpy.ops.object.camera_add(location=(0, -7.0, 3.0), rotation=(math.radians(62), 0, 0))
    camera = bpy.context.object
    camera.data.lens = 30
    camera.data.ortho_scale = max(5.0, total_width + 1.2)
    camera.data.type = "ORTHO"
    bpy.context.scene.camera = camera


def set_view_settings():
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64


def main():
    args = parse_args()
    clear_scene()

    objects = []
    labels = []
    for label, ply_path in args.item:
        obj = import_ply(ply_path)
        obj.name = label
        apply_vertex_color_material(obj)
        objects.append(obj)
        labels.append(label)

    total_width = normalize_and_place(objects, args.gap)
    for label, obj in zip(labels, objects):
        add_label(label, obj)
    add_camera_and_lights(total_width)
    set_view_settings()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
