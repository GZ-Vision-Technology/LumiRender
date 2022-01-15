import bpy
scene = bpy.data.scenes["Scene"]
#obj = bpy.context.scene.objects.matrix_world
# print(obj)
# for object in scene:
#    print(object.type)
#dir = bpy.path.abspath(bpy.data.scenes[0].exportpath)
#name =dir+'test.gltf'
# print(filepath)
# bpy.ops.export_scene.gltf(filepath=obj_filepath,
#                             export_format='GLTF_SEPARATE', export_materials='EXPORT', export_colors=True)
#textures = []
# for ob in bpy.data.objects:
# print(ob.name)
#    for mat_slot in ob.material_slots:
#        print(mat_slot.material)
#        for key,val in mat_slot.material:
#            print(val)
#            if mtex_slot:
#                # dump(mtex_slot)
#                print("\t%s" % mtex_slot)
#                if hasattr(mtex_slot.texture , 'image'):
#                    print("\t\t%s" % mtex_slot.texture.image.filepa)
# for ob in bpy.data.objects:
#    if ob.type == "MESH":
#        print('object name:',ob.name,'length:',len(ob.material_slots))
#        for slot_idx in range(len(ob.material_slots)):
#            mat = ob.material_slots[slot_idx].material
#            print ('Fetched material in slot 0 named: ', mat.name)
#            for node in mat.node_tree.nodes:
#                if node.name == 'Material Output':
#                    for input in node.inputs:
#                        for node_links in input.links:
#                            currentMaterial =  node_links.from_node
#                            print("Current mat id name:")
#                            print(currentMaterial.bl_idname)
#            for mtex_slot in mat_slot.material.texture_slots:
#                if mtex_slot:
#                    # dump(mtex_slot)
#                    print("\t%s" % mtex_slot)
#                    if hasattr(mtex_slot.texture , 'image'):
#                        print("\t\t%s" % mtex_slot.texture.image.filepath)


for ob in bpy.data.objects:
    if ob.type == "MESH":
        print('object name:', ob.name, 'length:', len(ob.material_slots))
        for slot_idx in range(len(ob.material_slots)):
            mat = ob.material_slots[slot_idx].material
            print('Fetched material in slot 0 named: ', mat.name)
            for node in mat.node_tree.nodes:
                if node.type == 'Image Texture Node':
                    print(node.name)
print('OK')
# print(textures)
