import bpy
from . import render_exporter

class ExportPbrtScene(bpy.types.Operator):
    bl_idname = 'scene.export'
    bl_label = 'Export Scene To Luminous Renderer'
    bl_options = {"REGISTER", "UNDO"}
    COMPAT_ENGINES = {'Luminous_Renderer'}
    
    def execute(self, context):
        print("Starting calling pbrt_export")
        print("Output path:")
        filepath_full = bpy.path.abspath(bpy.data.scenes[0].exportpath)
        print(filepath_full)
        # for frameNumber in range(bpy.data.scenes['Scene'].batch_frame_start, bpy.data.scenes['Scene'].batch_frame_end +1):
        #     bpy.data.scenes['Scene'].frame_set(frameNumber)
        #     print("Exporting frame: %s" % (frameNumber))
        #     render_exporter.export_luminous(filepath_full, bpy.data.scenes['Scene'], '{0:05d}'.format(frameNumber))
        render_exporter.export_luminous(filepath_full, bpy.data.scenes['Scene'])
        self.report({'INFO'}, "Export complete.")
        return {"FINISHED"}

class LuminousRenderSettingsPanel(bpy.types.Panel):
    """Creates a Luminous settings panel in the render context of the properties editor"""
    bl_label = "Luminous Render settings"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'Luminous_Renderer'}

    @classmethod
    def poll(cls, context):
        engine = context.scene.render.engine
        if engine != 'Luminous_Renderer':
            return False
        else:
            return True

    def draw(self, context):
        engine = context.scene.render.engine
        if engine != 'Luminous_Renderer':
            bpy.utils.unregister_class(LuminousRenderSettingsPanel)

        layout = self.layout

        scene = context.scene

        layout.label(text="Output folder path")
        row = layout.row()
        row.prop(scene, "exportpath")

        row = layout.row()
        row.prop(scene, "scene_name")

        layout.label(text=" Render output")
        row = layout.row()
        row.prop(scene,"outputfilename")
        row = layout.row()
        row.prop(scene, "frame_num")
        row = layout.row()
        row.prop(scene, "file_format")

        layout.label(text="Environment Map")
        row = layout.row()
        row.prop(scene,"environmentmaptpath")

        layout.label(text="Environment map scale:")
        row = layout.row()
        row.prop(scene, "environmentmapscale")

        layout.label(text="Film settings:")
        row = layout.row()
        row.prop(scene, "resolution_x")
        row.prop(scene, "resolution_y")
        row = layout.row()
        row.prop(scene, "fb_state")
       
        layout.label(text="Filter settings:")
        row = layout.row()
        row.prop(scene,"filterType")
        row = layout.row()
        row.prop(scene,"filter_x_width")
        row.prop(scene,"filter_y_width")

        if scene.filterType == 'sinc':
            row = layout.row()
            row.prop(scene,"filter_tau")
        if scene.filterType == 'mitchell':
            row = layout.row()
            row.prop(scene,"filter_b")
            row.prop(scene,"filter_c")
        if scene.filterType == 'gaussian':
            row = layout.row()
            row.prop(scene,"filter_alpha")
            
        layout.label(text="Integrator settings:")
        row = layout.row()

        row.prop(scene,"integrators")
        row.prop(scene,"maxdepth")
        row = layout.row()
        row.prop(scene,"rr_threshold")


        layout.label(text="Sampler settings:")
        row = layout.row()
        row.prop(scene,"sampler")
        row = layout.row()
        if scene.sampler == 'halton':
            row.prop(scene,"spp")
            row.prop(scene,"samplepixelcenter")
        if scene.sampler == 'PCGSampler':
            row.prop(scene,"spp")
        

        layout.label(text="Light strategy:")
        row = layout.row()
        row.prop(scene,"light_sampler")
        
        layout.label(text="Export:")
        row = layout.row()
        layout.operator("scene.export", icon='MESH_CUBE', text="Export scene")

def register():

    bpy.types.Scene.scene_name = bpy.props.StringProperty(
        name="",
        description="Export folder",
        default="luminous_scene",
        maxlen=1024,
        subtype='FILE_NAME'
    )

    bpy.types.Scene.exportpath = bpy.props.StringProperty(
        name="",
        description="Export folder",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')

    bpy.types.Scene.environmentmaptpath = bpy.props.StringProperty(
        name="",
        description="Environment map",
        default="",
        maxlen=1024,
        subtype='FILE_PATH')


    bpy.types.Scene.outputfilename = bpy.props.StringProperty(
        name="",
        description="Image output file name",
        default="output.png",
        maxlen=1024,
        subtype='FILE_NAME')

    bpy.types.Scene.frame_num = bpy.props.IntProperty(name = "frame num", description = "num of frame to save picture", 
                                                default = 100, min = 1, max = 9999)

    bpy.types.Scene.spp = bpy.props.IntProperty(name = "Samples per pixel", description = "Set spp", 
                                                default = 1, min = 1, max = 9999)
    bpy.types.Scene.maxdepth = bpy.props.IntProperty(name = "Max depth", description = "Set max depth", 
                                                default = 10, min = 1, max = 9999)

    integrators = [("PT", "PT", "", 1), ("wavefrontPT", "wavefrontPT", "", 2)]
    bpy.types.Scene.integrators = bpy.props.EnumProperty(name = "Name", items=integrators , default="PT")
    
    file_formats = [(".ply", ".ply", "", 1), (".gltf", ".gltf", "", 2)]
    bpy.types.Scene.file_format = bpy.props.EnumProperty(name = "Name", items=file_formats , default=".gltf")
    

    light_sampler = [("UniformLightSampler", "uniform", "", 1), 
                    ("PowerLightSampler", "power", "", 2), 
                    ("BVHLightSampler", "bvh", "", 3)]
    bpy.types.Scene.light_sampler = bpy.props.EnumProperty(name = "light_sampler", items=light_sampler , default="UniformLightSampler")

    bpy.types.Scene.environmentmapscale = bpy.props.FloatProperty(name = "Env. map scale", description = "Env. map scale", default = 1, min = 0.001, max = 9999)
    
    framebuffer_state = [("render", "render", "", 1), 
                    ("normal", "normal", "", 2), 
                    ("albedo", "albedo", "", 3)]
    bpy.types.Scene.fb_state = bpy.props.EnumProperty(name = "framebuffer_state", 
                    items=framebuffer_state , default="render")

    bpy.types.Scene.resolution_x = bpy.props.IntProperty(name = "X", description = "Resolution x", default = 768, min = 1, max = 9999)
    bpy.types.Scene.resolution_y = bpy.props.IntProperty(name = "Y", description = "Resolution y", default = 768, min = 1, max = 9999)

    bpy.types.Scene.lensradius = bpy.props.FloatProperty(name = "Lens radius", description = "Lens radius", default = 0, min = 0.001, max = 9999)
    
    bpy.types.Scene.batch_frame_start = bpy.props.IntProperty(name = "Frame start", description = "Frame start", 
                                                            default = 1, min = 1, max = 9999999)
    bpy.types.Scene.batch_frame_end = bpy.props.IntProperty(name = "Frame end", description = "Frame end", 
                                                            default = 1, min = 1, max = 9999999)

    bpy.types.Scene.rr_threshold = bpy.props.FloatProperty(name = "rr Threshold", description = "rr Threshold", 
                                                        default = 1.0, min = 0.001, max = 9999)

    filterTypes = [("BoxFilter", "BoxFilter", "", 1), 
                   ("GaussianFilter", "GaussianFilter", "", 2),
                   ("MitchellFilter", "MitchellFilter", "", 2),
                   ("LanczosSincFilter", "LanczosSincFilter", "", 4),
                   ("TriangleFilter", "TriangleFilter", "", 5)]
    bpy.types.Scene.filterType = bpy.props.EnumProperty(name = "filterType", items=filterTypes , default="triangle")
    bpy.types.Scene.filter_x_width = bpy.props.FloatProperty(name = "x", description = "x", default = 0.5, min = 0.0, max = 999)
    bpy.types.Scene.filter_y_width = bpy.props.FloatProperty(name = "y", description = "y", default = 0.5, min = 0.001, max = 999)
    bpy.types.Scene.filter_tau = bpy.props.FloatProperty(name = "tau", description = "tau", default = 3.0, min = 0.001, max = 999)
    bpy.types.Scene.filter_b = bpy.props.FloatProperty(name = "b", description = "b", default = 3.0, min = 0.0, max = 999)
    bpy.types.Scene.filter_c = bpy.props.FloatProperty(name = "c", description = "c", default = 3.0, min = 0.0, max = 999)
    bpy.types.Scene.filter_alpha = bpy.props.FloatProperty(name = "alpha", description = "alpha", default = 2.0, min = 0.0, max = 999)

    samplers = [("PCGSampler", "PCGSampler", "", 1), ("HaltonSampler", "HaltonSampler", "", 2)]
    bpy.types.Scene.sampler = bpy.props.EnumProperty(name = "Sampler", items=samplers , default="PCGSampler")
    bpy.types.Scene.samplepixelcenter = bpy.props.BoolProperty(name="sample pixel center", 
                                                            description="sample pixel center", default = False)
    bpy.types.Scene.dimension = bpy.props.IntProperty(name = "dimension", description = "dimension",
                                                     default = 4, min = 0, max = 9999999)
    bpy.types.Scene.jitter = bpy.props.BoolProperty(name="jitter", description="jitter", default = True)
    bpy.types.Scene.xsamples = bpy.props.IntProperty(name = "xsamples", description = "xsamples", default = 4, min = 0, max = 9999999)
    bpy.types.Scene.ysamples = bpy.props.IntProperty(name = "ysamples", description = "ysamples", default = 4, min = 0, max = 9999999)