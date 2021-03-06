
set(PUBLIC_HEADDER_DIRS "../../include/ganondorf")

set(src_files datacore.c arrayfunctions.c datasettype.c)

file(GLOB public_headder_files "${PUBLIC_HEADDER_DIRS}/*.h")
file(GLOB private_headder_files *.h)

#Headers are added so Visual studios sln looks nice
add_library(datacore 
  SHARED
  ${src_files} 
  ${public_headder_files} ${private_headder_files})

if (PYTHON_EXTENSION)
  message(STATUS "Setting library extension to ${PYTHON_EXTENSION}")

  set_target_properties(datacore 
    PROPERTIES 
    SUFFIX ${PYTHON_EXTENSION}
    PREFIX "")
endif()

target_include_directories(datacore 
  PUBLIC 
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
  ${PUBLIC_HEADDER_DIRS})

target_compile_definitions(datacore PUBLIC 
  MODULE_ARRAY_API_NAME=datacore_ARRAY_API)

target_link_libraries(datacore ${Python3_LIBRARIES})

#find_package(SimpleITK)
#add_executable(sitk_example sitk_example.cpp)
#target_link_libraries(sitk_example ${SimpleITK_LIBRARIES} )


if(BUILD_EXTENSIONS_INPLACE)
  message(STATUS "Build will copy C Extension inplace")
  inplace_build(datacore)
endif()


