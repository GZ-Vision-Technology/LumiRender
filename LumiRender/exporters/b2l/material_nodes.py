
import bpy
from bpy.types import NodeTree, Node, NodeSocket
import nodeitems_utils
from nodeitems_utils import (
    NodeCategory,
    NodeItem,
    NodeItemCustom,
)
# nodecategory begin


class MyNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        # Do not add the PBRT shader category if PBRT is not selected as renderer
        if context.space_data.tree_type == 'ShaderNodeTree':
            return True
        else:
            return False


# all categories in a list
node_categories = [
    # identifier, label, items list
    MyNodeCategory("B2L_SHADER", "B2L", items=[
        NodeItem("B2L_Node_Matte"),
        NodeItem("B2L_Node_FakeMetal"),
        NodeItem("B2L_Node_Disney"),
    ]),
]

# nodecategory end
# Derived from the NodeTree base type, similar to Menu, Operator, Panel, etc.


class MyCustomTree(NodeTree):
    bl_idname = 'B2L_TreeType'
    bl_label = 'Custom Node Tree'
    bl_icon = 'NODETREE'

# Defines a poll function to enable filtering for various node tree types.


class MyCustomTreeNode:
    bl_icon = 'INFO'

    @classmethod
    def poll(cls, ntree):
        b = False
        # Make your node appear in different node trees by adding their bl_idname type here.
        if ntree.bl_idname == 'ShaderNodeTree':
            b = True
        return b
    
    def draw_label(self):
        return self.bl_label

# Derived from the Node base type.


class B2L_Matte(Node, MyCustomTreeNode):
    '''A custom node'''
    bl_idname = 'B2L_Node_Matte'
    bl_label = 'Matte'
    bl_icon = 'INFO'

    def updateViewportColor(self, context):
        mat = bpy.context.active_object.active_material
        if mat is not None:
            bpy.data.materials[mat.name].diffuse_color = self.Kd

    Sigma: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    Kd: bpy.props.FloatVectorProperty(name="Kd", description="Kd", default=(
        0.8, 0.8, 0.8, 1.0), min=0, max=1, subtype='COLOR', size=4, update=updateViewportColor)

    def init(self, context):
        self.outputs.new('NodeSocketFloat', "B2L Matte")
        KdTexture_node = self.inputs.new('NodeSocketColor', "Kd Texture")
        KdTexture_node.hide_value = True

    def draw_buttons(self, context, layout):
        layout.prop(self, "Sigma", text='Sigma')
        layout.prop(self, "Kd", text='Kd')
    
class B2L_FakeMetal(Node, MyCustomTreeNode):
    '''A custom node'''
    bl_idname = 'B2L_Node_FakeMetal'
    bl_label = 'FakeMetal'
    bl_icon = 'INFO'
    
    def updateViewportColor(self, context):
        mat = bpy.context.active_object.active_material
        if mat is not None:
            bpy.data.materials[mat.name].diffuse_color = self.color

    color: bpy.props.FloatVectorProperty(name="color", description="color", default=(
        0.8, 0.8, 0.8, 1.0), min=0, max=1, subtype='COLOR', size=4, update=updateViewportColor)
    
    roughness: bpy.props.FloatProperty(default=0.1, min=0.0, max=1.0)

    def init(self, context):
        self.outputs.new('NodeSocketFloat', self.bl_label)
        colorTexture_node = self.inputs.new('NodeSocketColor', "color Texture")
        colorTexture_node.hide_value = True

    def draw_buttons(self, context, layout):
        layout.prop(self, "roughness", text='roughness')
        layout.prop(self, "color", text='color')

class B2L_Disney(Node, MyCustomTreeNode):
    '''A custom node'''
    bl_idname = 'B2L_Node_Disney'
    bl_label = 'Disney'
    bl_icon = 'INFO'
    
    def updateViewportColor(self, context):
        mat = bpy.context.active_object.active_material
        if mat is not None:
            bpy.data.materials[mat.name].diffuse_color = self.color

    color: bpy.props.FloatVectorProperty(name="color", description="color", default=(
        0.8, 0.8, 0.8, 1.0), min=0, max=1, subtype='COLOR', size=4, update=updateViewportColor)
    metallic: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    eta: bpy.props.FloatProperty(default=1.5, min=0.0, max=5)
    roughness: bpy.props.FloatProperty(default=1, min=0.0, max=1.0)
    specular_tint: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    anisotropic: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    sheen: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    sheen_tint: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    clearcoat: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    clearcoat_gloss: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    spec_trans: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    scatter_distance: bpy.props.FloatVectorProperty(name="color", description="color", default=(
        0, 0, 0, 0), min=0, max=1, subtype='COLOR', size=4, update=updateViewportColor)
    flatness: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    diff_trans: bpy.props.FloatProperty(default=0, min=0.0, max=1.0)
    thin: bpy.props.BoolProperty(default=False)
    
    def init(self, context):
        self.outputs.new('NodeSocketFloat', self.bl_label)
        colorTexture_node = self.inputs.new('NodeSocketColor', "color Texture")
        colorTexture_node.hide_value = True
    
    def draw_buttons(self, context, layout):
        layout.prop(self, "color", text='color')
        layout.prop(self, "metallic", text='metallic')
        layout.prop(self, "eta", text='eta')
        layout.prop(self, "roughness", text='roughness')
        layout.prop(self, "specular_tint", text='specular_tint')
        layout.prop(self, "anisotropic", text='anisotropic')
        layout.prop(self, "sheen", text='sheen')
        layout.prop(self, "sheen_tint", text='sheen_tint')
        layout.prop(self, "clearcoat", text='clearcoat')
        layout.prop(self, "clearcoat_gloss", text='clearcoat_gloss')
        layout.prop(self, "spec_trans", text='spec_trans')
        layout.prop(self, "scatter_distance", text='scatter_distance')
        layout.prop(self, "flatness", text='flatness')
        layout.prop(self, "diff_trans", text='diff_trans')
        layout.prop(self, "thin", text='thin')
    


def register():
    nodeitems_utils.register_node_categories("B2L_NODES", node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories("B2L_NODES")
