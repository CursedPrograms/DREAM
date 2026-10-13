"""
Runs inside Blender (started by make_rig.py, not by hand):
    blender -b --factory-startup --python blender_build_rig.py -- data.npz texture.png out.blend out.fbx

Builds DREAM's rigged FLAME head from make_rig.py's data: textured mesh (skin and eye materials),
expression shape keys,
armature with skin weights. Saves a self-contained .blend (texture packed) and an .fbx
(texture embedded).
"""

import sys

import bpy
import numpy as np
from mathutils import Vector

data_path, tex_path, blend_out, fbx_out = sys.argv[sys.argv.index("--") + 1:]
d = np.load(data_path)
verts, faces = d["verts"], d["faces"]
joints, parents, names = d["joints"], d["parents"], [str(n) for n in d["joint_names"]]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------- mesh + UVs
mesh = bpy.data.meshes.new("DREAM_head")
mesh.from_pydata(verts.tolist(), [], faces.tolist())
mesh.update()
uv = mesh.uv_layers.new(name="UVMap")
uv.data.foreach_set("uv", d["vt"][d["ft"]].reshape(-1))      # from_pydata keeps loops in face order
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

head = bpy.data.objects.new("DREAM_head", mesh)
scene.collection.objects.link(head)

# ---------------------------------------------------------------- materials
# Skin and eyes share the texture (the eyeballs have their own corner of it) but are separate
# materials, so the eyes can be glossy - and swapped or tweaked on their own.
image = bpy.data.images.load(tex_path)


def textured(name, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.location = (-400, 300)
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)


textured("DREAM_skin", 0.6)
textured("DREAM_eyes", 0.05)
mesh.polygons.foreach_set("material_index", d["eye_faces"].astype(np.int32))

# ---------------------------------------------------------------- shape keys
head.shape_key_add(name="Basis", from_mix=False)
for i, delta in enumerate(d["exp_deltas"]):
    # FLAME expression components are unit-variance, so +-3 covers the usual range
    sk = head.shape_key_add(name=f"exp_{i:02d}", from_mix=False)
    sk.data.foreach_set("co", (verts + delta).ravel())
    sk.slider_min, sk.slider_max = -3.0, 3.0
if np.abs(d["photo_delta"]).max() > 0:
    # her own resting face from the front photo, on by default (FLAME's neutral has the lids
    # wider open than hers); set it to 0 for FLAME's neutral
    sk = head.shape_key_add(name="photo_expression", from_mix=False)
    sk.data.foreach_set("co", (verts + d["photo_delta"]).ravel())
    sk.value = 1.0

# ---------------------------------------------------------------- armature
arm_data = bpy.data.armatures.new("DREAM_rig")
arm = bpy.data.objects.new("DREAM_rig", arm_data)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")

# Tails only orient the bones (FLAME joints are points): neck points up, jaw forward/down
# to the chin so rotating it on X opens the mouth, eyes look forward (-Y in Blender).
top = verts[:, 2].max()
tails = {
    "root": joints[1],
    "neck": np.array([joints[1][0], joints[1][1], top]),
    "jaw": joints[2] + np.array([0, -0.08, -0.04]),
    "eye.L": joints[3] + np.array([0, -0.02, 0]),
    "eye.R": joints[4] + np.array([0, -0.02, 0]),
}
bones = []
for i, name in enumerate(names):
    b = arm_data.edit_bones.new(name)
    b.head = Vector(joints[i])
    tail = tails[name]
    if np.linalg.norm(tail - joints[i]) < 1e-3:          # root sits on the neck joint in some fits
        tail = joints[i] + np.array([0, 0, 0.03])
    b.tail = Vector(tail)
    if parents[i] >= 0:
        b.parent = bones[parents[i]]
    bones.append(b)
bpy.ops.object.mode_set(mode="OBJECT")

for i, name in enumerate(names):
    vg = head.vertex_groups.new(name=name)
    for vi in np.nonzero(d["weights"][:, i] > 1e-4)[0]:
        vg.add([int(vi)], float(d["weights"][vi, i]), "REPLACE")
# FLAME's neck runs almost to the shoulders: trim it. Sliced along a plane, so the edge is a clean
# line rather than the staircase of whole triangles; the new vertices on it get their UVs, bone
# weights and shape keys interpolated from the edges they split.
import bmesh
bm = bmesh.new()
bm.from_mesh(mesh)
bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-6,
                       plane_co=(0.0, 0.0, float(d["neck_cut"])), plane_no=(0.0, 0.0, 1.0), clear_inner=True)
bm.to_mesh(mesh)
bm.free()

head.parent = arm
mod = head.modifiers.new("Armature", "ARMATURE")
mod.object = arm

# ---------------------------------------------------------------- save
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
bpy.ops.export_scene.fbx(filepath=fbx_out, use_selection=False, object_types={"ARMATURE", "MESH"},
                         add_leaf_bones=False, path_mode="COPY", embed_textures=True,
                         mesh_smooth_type="FACE")
print(f"saved {blend_out} and {fbx_out}")

# ---------------------------------------------------------------- preview
# Front, 3/4, side and back renders for make_rig.py to put side by side: soft even light plus a
# key light, so the skin shows its real colour. Rendered after saving so the camera and lights
# never end up in the files.
import math
centre = Vector(verts.mean(0).tolist())
cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
scene.collection.objects.link(cam)
scene.camera = cam
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 32
scene.world = bpy.data.worlds.new("preview_world")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.82, 0.82, 0.82, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.75
key = bpy.data.objects.new("preview_key", bpy.data.lights.new("preview_key", "SUN"))
key.data.energy = 2.5
key.rotation_euler = (math.radians(50), 0, math.radians(-25))
scene.collection.objects.link(key)
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = scene.render.resolution_y = 800
for i, angle in enumerate((0, 45, 90, 180)):
    a = math.radians(angle)
    cam.location = centre + Vector((0.62 * math.sin(a), -0.62 * math.cos(a), 0.05))
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = blend_out[:-len(".blend")] + f"_view{i}.png"
    bpy.ops.render.render(write_still=True)
