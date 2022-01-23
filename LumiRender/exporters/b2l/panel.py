# -*- coding:utf-8 -*-

import bpy
from . import exporter


class SidebarSetup:
    bl_category = "blender2luminous"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return True


class B2L_PT_Base_Panel(SidebarSetup, bpy.types.Panel):
    bl_label = "Base Setting"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout.box()
        scene = context.scene
        split = layout.split(factor=0.25)
        col_1 = split.column()
        col_2 = split.column()
        col_1.label(text="Path")
        col_2.prop(scene, "exportpath")

        col_1.label(text="Name")
        col_2.prop(scene, "outputfilename")
        
        picture_name_row = layout.row()
        picture_name_row.label(text="Picture Name")
        picture_name_row.prop(scene, "picture_name")
        
        dispatch_num_row = layout.row()
        dispatch_num_row.label(text="dispatch num")
        dispatch_num_row.prop(scene, "dispatch_num")
        
        layout = self.layout.box()
        
        scene = context.scene
        row = layout.row()
        row.label(text="Mode")
        row.prop(scene, "rendermode",  expand=True)
        row = layout.row()
        # row.label(text="Mode")
        row.prop(scene, "meshtype",  expand=True)

        layout = self.layout.box()
        row = layout.row()
        layout.scale_y = 2.0
        layout.operator("b2l.import_test_scene",
                        icon="COMMUNITY", text="import test scene !")
        layout.operator("b2l.export_scene", icon="MESH_CUBE", text="Export !")


class B2L_PT_Environment_Panel(SidebarSetup, bpy.types.Panel):
    bl_label = "EnvironmentMap Setting"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout.box()
        split = layout.split(factor=0.25)
        scene = context.scene

        col_1 = split.column()
        col_2 = split.column()
        col_1.label(text="Path")
        col_2.prop(scene, "environmentmaptpath")

        col_1.label(text="Scale")
        col_2.prop(scene, "environmentmapscale")


class B2L_PT_Other_Panel(SidebarSetup, bpy.types.Panel):
    bl_label = "Other Setting"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        scene = context.scene
        layout = self.layout.box()
        row = layout.row()
        row.prop(scene, "sampler")
        row = layout.row()
        row.prop(scene, "spp")

        layout = self.layout.box()
        row = layout.row()
        row.prop(scene, "integrator")
        row = layout.row()
        row.prop(scene, "max_depth")
        row.prop(scene, "min_depth")
        row.prop(scene, "rr_threshold")
        layout = self.layout.box()
        row = layout.row()
        row.prop(scene, "lightscale")

# more mode


class B2L_PT_Camera_Panel(SidebarSetup, bpy.types.Panel):
    bl_label = "Camera Setting"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        scene = context.scene
        layout = self.layout.box()
        row = layout.row()
        row.label(text="Camera Type")
        row.prop(scene, "cameratpye", expand=True)
        row = layout.row()
        row.label(text="Resolution")
        row.prop(scene, "resolution_x")
        row.prop(scene, "resolution_y")

        layout = self.layout.box()
        split = layout.split(factor=0.25)
        col_1 = split.column()
        col_2 = split.column()

        col_1.label(text="Velocity")
        col_2.prop(scene, "cameravelocity")

        layout = self.layout.box()
        row = layout.row()
        row.prop(scene, "filterType")
        row = layout.row()
        row.prop(scene, "filter_radius_x")
        row.prop(scene, "filter_radius_y")
        if scene.filterType == 'GaussianFilter':
            row = layout.row()
            row.prop(scene, "filter_sigma")
        elif scene.filterType == "LanczosSincFilter":
            row = layout.row()
            row.prop(scene, "filter_tau")
        elif scene.filterType == "MitchellFilter":
            row = layout.row()
            row.prop(scene, "filter_B")
            row.prop(scene, "filter_C")

        # more mode


class B2L_OT_import_test_scene(bpy.types.Operator):
    bl_idname = "b2l.import_test_scene"
    bl_label = "import_test_scene"
    # bl_options = {"REGISTER", "UNDO"}
    # COMPAT_ENGINES = {"Luminous_Renderer"}

    def execute(self, context):
        bpy.ops.wm.open_mainfile(
            filepath="samplescene/cornell_box.blend")
        return {"FINISHED"}


class B2L_OT_export_scene(bpy.types.Operator):
    bl_idname = "b2l.export_scene"
    bl_label = "Export Scene To Luminous Renderer"
    # bl_options = {"REGISTER", "UNDO"}
    # COMPAT_ENGINES = {"Luminous_Renderer"}

    def execute(self, context):
        print("Starting calling exporter")
        print("Output path:")
        filepath_full = bpy.path.abspath(bpy.data.scenes[0].exportpath)
        print(filepath_full)
        # for frameNumber in range(bpy.data.scenes['Scene'].batch_frame_start, bpy.data.scenes['Scene'].batch_frame_end +1):
        #     bpy.data.scenes['Scene'].frame_set(frameNumber)
        #     print("Exporting frame: %s" % (frameNumber))
        #     render_exporter.export_luminous(filepath_full, bpy.data.scenes['Scene'], '{0:05d}'.format(frameNumber))
        # exporter2.export_luminous(bpy.data.scenes["Scene"])
        exporter.export_test(bpy.data.scenes["Scene"], filepath_full)
        self.report({"INFO"}, "Export complete.")
        return {"FINISHED"}
# Assign a collection


class SceneSettingItem(bpy.types.PropertyGroup):
    # light_sampler_type

    filterTypes = [("GaussianFilter", "GaussianFilter", "", 1),
                   ("BoxFilter", "BoxFilter", "", 2),
                   ("TriangleFilter", "TriangleFilter", "", 3),
                   ("LanczosSincFilter", "LanczosSincFilter", "", 4),
                   ("MitchellFilter", "MitchellFilter", "", 5)]

    rendermodes = [('render', 'render', ''),
                   ('normal', 'normal', ''),
                   ('albedo', 'albedo', '')]
    integrators = [
        ("PT", "PT", "", 1),
        ("WavefrontPT", "WavefrontPT", "", 2),
    ]
    samplers = [("PCGSampler", "PCGSampler", ""),
                ("LCGSampler", "LCGSampler", "")]
    out_meshes_type = [("Single", "Single", ""),
                       ("All", "All", "")]

    bpy.types.Scene.filterType = bpy.props.EnumProperty(
        name="Filter", items=filterTypes, default="GaussianFilter")
    bpy.types.Scene.exportpath = bpy.props.StringProperty(
        name="",
        description="Export folder",
        default="output",
        maxlen=1024,
        subtype="DIR_PATH",
    )
    bpy.types.Scene.outputfilename = bpy.props.StringProperty(
        name="",
        description="json output file name",
        default="scene.json",
        maxlen=1024,
        subtype='FILE_NAME')
    
    bpy.types.Scene.picture_name = bpy.props.StringProperty(
        name="",
        description="output render result",
        default="scene.png",
        maxlen=1024,
        subtype='FILE_NAME')
    
    bpy.types.Scene.dispatch_num = bpy.props.IntProperty(
        name="", description="after dispatch num render output", default=0, min=0)

    bpy.types.Scene.rendermode = bpy.props.EnumProperty(
        name="rendermode", items=rendermodes, default="render")

    bpy.types.Scene.environmentmaptpath = bpy.props.StringProperty(
        name="",
        description="Environment map",
        default="",
        maxlen=1024,
        subtype='FILE_PATH')
    bpy.types.Scene.environmentmapscale = bpy.props.FloatProperty(
        name="", description="Env. map scale", default=1, min=0.001, max=9999)

    bpy.types.Scene.sampler = bpy.props.EnumProperty(
        name="Sampler", items=samplers, default="PCGSampler")
    bpy.types.Scene.meshtype = bpy.props.EnumProperty(
        name="Mesh Type", items=out_meshes_type, default="Single")
    cameratpyes = [("ThinLensCamera", "ThinLensCamera", ""),
                   ("RealisticCamera", "RealisticCamera", "")]
    bpy.types.Scene.cameratpye = bpy.props.EnumProperty(
        name="Camera Tpye", items=cameratpyes, default="ThinLensCamera")

    bpy.types.Scene.cameravelocity = bpy.props.IntProperty(
        name="", description="velocity", default=20, min=1, max=9999)

    bpy.types.Scene.resolution_x = bpy.props.IntProperty(
        name="X", description="Resolution x", default=1024, min=1, max=9999)
    bpy.types.Scene.resolution_y = bpy.props.IntProperty(
        name="Y", description="Resolution y", default=768, min=1, max=9999)

    # filter setting
    bpy.types.Scene.filter_radius_x = bpy.props.FloatProperty(
        name="filter_radius_x", description="x", default=2, min=0.0, max=5)
    bpy.types.Scene.filter_radius_y = bpy.props.FloatProperty(
        name="filter_radius_y", description="y", default=2, min=0.0, max=5)
    bpy.types.Scene.filter_sigma = bpy.props.FloatProperty(
        name="filter_sigma", description="filter_sigma", default=0.5, min=0.0, max=10)
    bpy.types.Scene.filter_tau = bpy.props.FloatProperty(
        name="tau", description="sinc filter tau", default=3, min=0.0, max=10)
    bpy.types.Scene.filter_B = bpy.props.FloatProperty(
        name="B", description="mitchell filter B", default=0.333, min=0.0, max=10)
    bpy.types.Scene.filter_C = bpy.props.FloatProperty(
        name="C", description="mitchell filter C", default=0.333, min=0.0, max=10)

    # sampler setting
    bpy.types.Scene.spp = bpy.props.IntProperty(
        name="spp", description="Set Samples per pixel", default=1, min=1, max=9999)

    # integrator setting
    bpy.types.Scene.integrator = bpy.props.EnumProperty(
        name="integrator", items=integrators, default="PT")
    bpy.types.Scene.min_depth = bpy.props.IntProperty(
        name="min_depth", description="Set min depth", default=0, min=0, max=10)
    bpy.types.Scene.max_depth = bpy.props.IntProperty(
        name="max_depth", description="Set max depth", default=10, min=1, max=999)
    bpy.types.Scene.rr_threshold = bpy.props.IntProperty(
        name="rr_threshold", description="rr_threshold", default=1, min=0, max=1)

    # light setting
    bpy.types.Scene.lightscale = bpy.props.FloatProperty(
        name="Light Scale", description="convert to engine light", default=1, min=0, max=10)


def register():

    bpy.types.Scene.my_settings = bpy.props.CollectionProperty(
        type=SceneSettingItem)


def unregister():
    del bpy.types.Scene.my_settings


if __name__ == '__main__':
    register()
