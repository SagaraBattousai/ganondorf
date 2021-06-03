
set(PUBLIC_HEADDER_DIRS "../../include/ganondorf")

set(src_files datamodule.c arrayfunctions.c datasettype.c)

file(GLOB public_headder_files "${PUBLIC_HEADDER_DIRS}/*.h")
file(GLOB private_headder_files *.h)

#Headers are added so Visual studios sln looks nice
add_library(datamodule 
  SHARED
  ${src_files} 
  ${public_headder_files} ${private_headder_files})

if (PYTHON_EXTENSION)
  message(STATUS "Setting library extension to ${PYTHON_EXTENSION}")

  set_target_properties(datamodule PROPERTIES SUFFIX ${PYTHON_EXTENSION})
endif()

target_include_directories(datamodule 
  PUBLIC 
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${PUBLIC_HEADDER_DIRS})


target_compile_definitions(datamodule PUBLIC 
  MODULE_ARRAY_API_NAME=datamodule_ARRAY_API)

message(STATUS ${Python_NumPy_INCLUDE_DIRS})

target_link_libraries(datamodule ${Python_LIBRARIES})

#find_package(SimpleITK)
#add_executable(sitk_example sitk_example.cpp)
#target_link_libraries(sitk_example ${SimpleITK_LIBRARIES} )


if(BUILD_EXTENSIONS_INPLACE)
  message(STATUS "Build will copy C Extension inplace")
  inplace_build(datamodule)
endif()


