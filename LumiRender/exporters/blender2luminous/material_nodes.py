#NOTE: Run this code first then use SHIFT-A, below, to add Custom Float node type.

import bpy
from bpy.types import NodeTree, Node, NodeSocket
import nodeitems_utils
from nodeitems_utils import (
    NodeCategory,
    NodeItem,
    NodeItemCustom,
)


#nodecategory begin
class MyNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        #Do not add the Luminous shader category if PBRT is not selected as renderer
        engine = context.scene.render.engine
        if engine != 'Luminous_Renderer':
            return False
        else:
            b = False
            if context.space_data.tree_type == 'ShaderNodeTree': b = True
            return b

# all categories in a list
node_categories = [
    # identifier, label, items list
    MyNodeCategory("Luminous_SHADER", "Luminous", items=[
        NodeItem("CustomNodeTypeMatte"),
        ]),
    ]

#nodecategory end


# Implementation of custom nodes from Python
# Derived from the NodeTree base type, similar to Menu, Operator, Panel, etc.
class MyCustomTree(NodeTree):
    bl_idname = 'CustomTreeType'
    bl_label = 'Custom Node Tree'
    bl_icon = 'NODETREE'

# Defines a poll function to enable filtering for various node tree types.
class MyCustomTreeNode :
    bl_icon = 'INFO'
    @classmethod
    def poll(cls, ntree):
        b = False
        # Make your node appear in different node trees by adding their bl_idname type here.
        if ntree.bl_idname == 'ShaderNodeTree': b = True
        return b

# Derived from the Node base type.
class LuminousMatte(Node, MyCustomTreeNode):
    '''A custom node'''
    bl_idname = 'CustomNodeTypeMatte'
    bl_label = 'Luminous Matte'
    bl_icon = 'INFO'

    def updateViewportColor(self,context):
        mat = bpy.context.active_object.active_material
        if mat is not None:
            bpy.data.materials[mat.name].diffuse_color=self.Kd
        
    Sigma : bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    Kd : bpy.props.FloatVectorProperty(name="Kd", description="Kd",default=(0.8, 0.8, 0.8, 1.0), min=0, max=1, subtype='COLOR', size=4,update=updateViewportColor)
    
    def init(self, context):
        self.outputs.new('NodeSocketFloat', "Luminous Matte")
        KdTexture_node = self.inputs.new('NodeSocketColor', "Kd Texture")
        KdTexture_node.hide_value = True

    def draw_buttons(self, context, layout):
        layout.prop(self, "Sigma",text = 'Sigma')
        layout.prop(self, "Kd",text = 'Kd')
        
    def draw_label(self):
        return "Luminous Matte"

#@base.register_class
class PbrtTextureSocket(bpy.types.NodeSocket):
    bl_idname = 'PbrtTextureSocket'
    bl_label = 'Pbrt Texture Socket'

    default_color : bpy.props.FloatVectorProperty(
        name='Color',
        description='Color',
        subtype='COLOR',
        min=0.0,
        soft_max=1.0,
        default=(0.8, 0.8, 0.8),
    )

    default_value : bpy.props.FloatProperty(
        name='Value',
        description='Value',
        min=0.0,
        soft_max=1.0,
        default=0.5,
    )

    tex_type : bpy.props.EnumProperty(
        name='Texture Type',
        description='Texture Type',
        items=[
            ('COLOR', 'Color', ''),
            ('VALUE', 'Value', ''),
            ('PURE', 'Pure', ''),
        ],
        default='COLOR',
    )

    def to_scene_data(self, scene):
        if self.is_linked:
            d = self.links[0].from_node.to_scene_data(scene)
            if d:
                return d
        if self.tex_type == 'COLOR':
            return list(self.default_color)
        elif self.tex_type == 'VALUE':
            return self.default_value
        else:
            return 0.0

    def draw_value(self, context, layout, node):
        layout.label(self.name)

    def draw_color(self, context, node):
        return (1.0, 0.1, 0.2, 0.8)

    def draw(self, context, layout, node, text):
        if self.tex_type == 'PURE' or self.is_output or self.is_linked:
            layout.label(self.name)
        else:
            if self.tex_type == 'COLOR':
                layout.prop(self, 'default_color', text=self.name)
            elif self.tex_type == 'VALUE':
                layout.prop(self, 'default_value', text=self.name)


def register():
    nodeitems_utils.register_node_categories("CUSTOM_NODES", node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories("CUSTOM_NODES")
