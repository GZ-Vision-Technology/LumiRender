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

        layout = self.layout.box()
        scene = context.scene
        row = layout.row()
        row.label(text="Mode")
        row.prop(scene, "rendermode",  expand=True)

        layout = self.layout.box()
        row = layout.row()
        layout.scale_y = 2.0
        layout.operator("b2l.import_test_scene",
                        icon="COMMUNITY", text="import test scene !")
        layout.operator("b2l.export_scene", icon="MESH_CUBE", text="Export !")


class B2L_PT_Environment_Panel(SidebarSetup, bpy.types.Panel):
    bl_label = "Environment"

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
        if scene.sampler == 'PCGSampler':
            row.prop(scene, "spp")

        layout = self.layout.box()
        row = layout.row()
        row.prop(scene, "integrator")
        row = layout.row()
        if scene.integrator == 'PT':
            row.prop(scene, "max_depth")
            row.prop(scene, "rr_threshold")

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

        # col_1.label(text="Fov")
        # col_2.prop(scene, "cameratfov")
        col_1.label(text="Velocity")
        col_2.prop(scene, "cameravelocity")
        # col_1.label(text="Focal_distance")
        # # col_2.prop(scene, "focal_distance")
        # col_1.label(text="Lens_radius")
        # col_2.prop(scene, "lens_radius")

        layout = self.layout.box()
        row = layout.row()
        row.label(text="Filter")
        row.prop(scene, "filterType")
        # split = layout.split(factor=0.25)
        # col_1 = split.column()
        # col_2 = split.column()
        if scene.filterType == 'GaussianFilter':
            # col_1.label(text="radius_x")
            row = layout.row()
            row.prop(scene, "filter_radius_x")
            # col_1.label(text="radius_y")
            row.prop(scene, "filter_radius_y")
            # col_1.label(text="radius_y")
            row.prop(scene, "filter_sigma")

        # more mode


class B2L_OT_import_test_scene(bpy.types.Operator):
    bl_idname = "b2l.import_test_scene"
    bl_label = "import_test_scene"
    # bl_options = {"REGISTER", "UNDO"}
    # COMPAT_ENGINES = {"Luminous_Renderer"}

    def execute(self, context):
        bpy.ops.wm.open_mainfile(
            filepath="D:/code/blender2luminous/assets/scenes_all2.blend")
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
        exporter.export_test(bpy.data.scenes["Scene"])
        self.report({"INFO"}, "Export complete.")
        return {"FINISHED"}


def register():

    # light_sampler_type

    filterTypes = [("GaussianFilter", "GaussianFilter", "", 1),
                   ("Triangle", "Triangle", "", 2),
                   ("LanczosSincFilter", "LanczosSincFilter", "", 3),
                   ("Mitchell", "Mitchell", "", 4)]
    bpy.types.Scene.filterType = bpy.props.EnumProperty(
        name="", items=filterTypes, default="GaussianFilter")
    bpy.types.Scene.exportpath = bpy.props.StringProperty(
        name="",
        description="Export folder",
        default="D:/code/blender2luminous/assets",
        maxlen=1024,
        subtype="DIR_PATH",
    )
    bpy.types.Scene.outputfilename = bpy.props.StringProperty(
        name="",
        description="json output file name",
        default="output.json",
        maxlen=1024,
        subtype='FILE_NAME')

    rendermodes = [('render', 'render', ''),
                   ('normal', 'normal', ''),
                   ('albedo', 'albedo', '')]
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

    samplers = [("PCGSampler", "PCGSampler", ""),
                ("LCGSampler", "LCGSampler", "")]

    bpy.types.Scene.sampler = bpy.props.EnumProperty(
        name="Sampler", items=samplers, default="PCGSampler")
    cameratpyes = [("ThinLensCamera", "ThinLensCamera", ""),
                   ("Others", "Others", "")]
    bpy.types.Scene.cameratpye = bpy.props.EnumProperty(
        name="Camera Tpye", items=cameratpyes, default="ThinLensCamera")

    # bpy.types.Scene.cameratfov = bpy.props.IntProperty(
    #     name="", description="fov", default=35, min=1, max=9999)
    bpy.types.Scene.cameravelocity = bpy.props.IntProperty(
        name="", description="velocity", default=20, min=1, max=9999)
    # bpy.types.Scene.focal_distance = bpy.props.IntProperty(
    #     name="", description="focal_distance", default=35, min=1, max=9999)
    # bpy.types.Scene.cameravelocity = bpy.props.IntProperty(
    #     name="", description="velocity", default=20, min=1, max=9999)

    bpy.types.Scene.resolution_x = bpy.props.IntProperty(
        name="X", description="Resolution x", default=1024, min=1, max=9999)
    bpy.types.Scene.resolution_y = bpy.props.IntProperty(
        name="Y", description="Resolution y", default=768, min=1, max=9999)

    bpy.types.Scene.filter_radius_x = bpy.props.FloatProperty(
        name="filter_radius_x", description="x", default=3, min=0.0, max=999)
    bpy.types.Scene.filter_radius_y = bpy.props.FloatProperty(
        name="filter_radius_y", description="y", default=3, min=0.0, max=999)
    bpy.types.Scene.filter_sigma = bpy.props.FloatProperty(
        name="filter_sigma", description="filter_sigma", default=0.5, min=0.0, max=999)
    # bpy.types.Scene.filter_radius_y = bpy.props.FloatProperty(
    #     name="radius_Y", description="y", default=3, min=0.0, max=999)

    bpy.types.Scene.spp = bpy.props.IntProperty(
        name="Samples per pixel", description="Set spp", default=1, min=1, max=9999)

    integrators = [
        ("path", "path", "", 1), ("volpath", "volpath", "", 2),
        ("bdpt", "bdpt", "", 3), ("mlt", "mlt", "", 4),
        ("sppm", "sppm", "", 5), ("PT", "PT", "", 6)]

    bpy.types.Scene.integrator = bpy.props.EnumProperty(
        name="integrator", items=integrators, default="PT")

    bpy.types.Scene.max_depth = bpy.props.IntProperty(
        name="max_depth", description="Set max depth", default=3, min=1, max=9999)
    bpy.types.Scene.rr_threshold = bpy.props.IntProperty(
        name="rr_threshold", description="rr_threshold", default=1, min=1, max=9999)


if __name__ == '__main__':
    register()
