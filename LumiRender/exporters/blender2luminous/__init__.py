bl_info = {
    "name": "Blender Addon Example",
    "author": "babyformula",
    "version": (2021, 7, 30),
    "blender": (2, 80, 0),
    "location": "Viewport > Right panel",
    "description": "Blender Addon Example",
    "category": "Example"}

import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty )
from bpy.types import ( PropertyGroup )

import numpy as np
import os

class exampleLoadPose(bpy.types.Operator, ImportHelper):
    bl_idname = "object.example_load_pose"
    bl_label = "Load Pose"
    bl_description = ("Load pose from file")
    bl_options = {'REGISTER', 'UNDO'}
    
    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'}
    )

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return (context.object.type == 'MESH') or (context.object.type == 'ARMATURE')
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj # mesh needs to be active object for recalculating joint locations
        
        print("Loading: " + self.filepath)
        data = np.load(self.filepath)
        if "transl" in data:
            translation = np.array(data["transl"]).reshape(3)
        # Set translation
        if translation is not None:
            obj.location = (translation[0], -translation[2], translation[1])
            
        return {'FINISHED'}

class exampleExportPose(bpy.types.Operator, ExportHelper):
    bl_idname = "object.example_export_pose"
    bl_label = "Export Pose"
    bl_description = ("Export pose to Numpy Compressed NPZ format")
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper class uses this
    filename_ext = ".npz"
    
    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return (context.object.type == 'MESH') or (context.object.type == 'ARMATURE')
        except: return False
        
    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj # mesh needs to be active object for recalculating joint locations
            
        ret = {"transl": np.array(obj.location)}
        print("Exported: " + self.filepath)
        np.savez_compressed(self.filepath, **ret)
        
        return {'FINISHED'}
        
class examplePTLoad(bpy.types.Panel):
    bl_label = "Load"
    bl_category = "Example"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Load Pose:")
        col.operator("object.example_load_pose")
        col.separator()

class examplePTExport(bpy.types.Panel):
    bl_label = "Export"
    bl_category = "Example"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.label(text="Export Pose:")
        col.operator("object.example_export_pose")
        col.separator()

        row = col.row(align=True)
        row.operator("ed.undo", icon='LOOP_BACK')
        row.operator("ed.redo", icon='LOOP_FORWARDS')
        col.separator()

        (year, month, day) = bl_info["version"]
        col.label(text="Version: %s-%s-%s" % (year, month, day))
        
classes = [
    exampleLoadPose,
    exampleExportPose,
    examplePTLoad,
    examplePTExport
]

# ===================================================================

def register():
    from bpy.utils import register_class
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
