
function(inplace_build target)
  set(in_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  #get_filename_component(directory ${in_dir} NAME)
  get_filename_component(base_dir ${in_dir}/../.. ABSOLUTE)

  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E 
      copy_if_different
        "$<TARGET_FILE:${target}>"
        "${base_dir}/$<TARGET_FILE_BASE_NAME:${target}>${PYTHON_EXTENSION}"
    COMMAND ${CMAKE_COMMAND} -E 
      echo 
        "Copying ${target} to ${base_dir} for inplace build"
    VERBATIM)
endfunction()


#ganondorf\core\src\data -> ganondorf\core\data
