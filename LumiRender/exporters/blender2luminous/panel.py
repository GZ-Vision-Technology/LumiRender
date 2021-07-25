# -*- coding:utf-8 -*-

import bpy

import bpy

# 1. Create class
class TestPanel(bpy.types.Panel):
    bl_label="sdfasdfsd"
    bl_idname="PT_Test Panel"
    bl_space_type="VIEW_3D"
    bl_region_type="UI"
    bl_category="asdasd" #SideBar Name
    
    
   

# 2.Draw UI Function
    
    def draw(self,context):
        layout=self.layout
        
        # Text Show
        row=layout.row()    #Create a new row
        row.label(text='Add a girl',icon= 'BRUSH_MIX')
        row.label(text='Add a boy',icon= 'FUND')
        
        # Button Event
        row=layout.row()
        row.operator("mesh.primitive_cube_add",icon="LIGHTPROBE_CUBEMAP")
        
        
          

# 3.Update the Modified Class        
         
def register():
    bpy.utils.register_class(TestPanel)

def unregister():
    bpy.utils.unregister_class(TestPanel)
    
if __name__=="__main__":
    register()