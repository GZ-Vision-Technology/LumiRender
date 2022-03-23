

function(copy_assets)
  set(oneValueArgs OUTPUTS OUTPUT_DIRECTORY)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(copy "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  foreach(asset ${copy_SOURCES})
    if(IS_ABSOLUTE ${asset})
      get_filename_component(file_name ${asset} NAME)
    else()
      set(file_name ${asset})
    endif()
    get_filename_component(full_path ${asset} ABSOLUTE)
    set(output_file ${copy_OUTPUT_DIRECTORY}/${file_name})
    set(${copy_OUTPUTS} ${${copy_OUTPUTS}} ${output_file})
    set(${copy_OUTPUTS} ${${copy_OUTPUTS}} PARENT_SCOPE)
    set_source_files_properties(${asset} PROPERTIES HEADER_FILE_ONLY TRUE)
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${full_path} ${output_file}
      DEPENDS ${full_path}
    )
    # if (WIN32)
    #   add_custom_command(
    #     OUTPUT ${output_file}
    #     #COMMAND mklink \"${output_file}\" \"${full_path}\"
    #     COMMAND xcopy \"${full_path}\" \"${output_file}*\" /Y /Q /F
    #     DEPENDS ${full_path}
    #   )
    # else()
    #   add_custom_command(
    #     OUTPUT ${output_file}
    #     COMMAND mkdir --parents ${output_dir} && cp --force --link \"${full_path}\" \"${output_file}\"
    #     DEPENDS ${full_path}
    #   )
    # endif()
  endforeach()
endfunction(copy_assets)


set(ASSET_FILES
  cloth_specular_albedo/cloth_specular_albedo.jpg
  cloth_specular_albedo/cloth_specluar_albedo_average.jpg
)

copy_assets(
  SOURCES ${ASSET_FILES}
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/asset/"
  OUTPUTS copied_asset_files)

add_custom_target(CopyAssetFilesTarget ALL
  DEPENDS ${copied_asset_files}
  SOURCES ${ASSET_FILES}
)

source_group("asset" FILES ${ASSET_FILES})
