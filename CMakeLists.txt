# Example cmake command to build TALSH
# -DTALSH_GPU=ON -DTALSH_GPU_ARCH=70 
# (Optional) -DBUILD_TEST=ON -DWITH_CUTENSOR=ON -DCUTENSOR_ROOT=/cutensor_root -DTALSH_FINE_TIMING=ON

cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project (TALSH VERSION 0.1.0 LANGUAGES C CXX Fortran)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)

# find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

#TODO:
#Cray, Clang compiler flags
# DF := -WF, when using IBM compilers

if (TALSH_GPU)
	set(TALSH_GPU_DEFS "CUDA_ARCH=${TALSH_GPU_ARCH}0")
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_EXTENSIONS OFF)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
	set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
	find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
else()
	set(TALSH_GPU_DEFS NO_GPU)
endif()

if(WITH_CUTENSOR)
    set(TALSH_GPU_DEFS USE_CUTENSOR)
endif()

if(EXATN_SERVICE)
	set(TALSH_GPU_DEFS EXATN_SERVICE)
endif()

if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(TALSH_Fortran_FLAGS ${TALSH_Fortran_FLAGS} -fpp -vec-threshold4 -mkl=parallel)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
  set(TALSH_Fortran_FLAGS ${TALSH_Fortran_FLAGS} -Mcache_align -Mstandard -Mallocatable=03)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
  set(TALSH_Fortran_FLAGS ${TALSH_Fortran_FLAGS} -qpic=large)
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(TALSH_Fortran_FLAGS ${TALSH_FORTRAN_FLAGS} -hpic)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  set(TALSH_CXX_FLAGS ${TALSH_CXX_FLAGS} -silent -w -Mnovect)
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
#   set(TALSH_CXX_FLAGS ${TALSH_CXX_FLAGS} -h)
endif()

if (NOT TALSH_BLAS)
    set(NO_BLAS NO_BLAS)
endif()

set(TALSH_CXX_COMPILE_DEFS ${TALSH_GPU_DEFS} ${NO_BLAS} NO_AMD NO_PHI LINUX)
set(TALSH_Fortran_COMPILE_DEFS ${TALSH_GPU_DEFS} ${NO_BLAS} NO_AMD NO_PHI LINUX)

if (TALSH_GPU)
	set(TALSH_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
	if(WITH_CUTENSOR)
		list(APPEND TALSH_INCLUDES ${CUTENSOR_ROOT}/include)
	endif()
# else()
    # set(TALSH_INCLUDES ${MPI_C_INCLUDE_PATH} ${MPI_Fortran_INCLUDE_PATH})
endif()

set (TALSH_FORTRAN_SOURCES
	 dil_basic.F90
	 stsubs.F90
	 combinatoric.F90
	 symm_index.F90
	 timers.F90
	 tensor_algebra.F90
	 tensor_algebra_cpu.F90
	 tensor_algebra_cpu_phi.F90
	 tensor_dil_omp.F90
	 talshf.F90
	 )

set (TALSH_CXX_SOURCES
	timer.cpp
	byte_packet.cpp
	mem_manager.cpp
	talshc.cpp
	talsh_task.cpp
	talshxx.cpp
    )

if(TALSH_GPU)
	set(TALSH_CUDA_SOURCES tensor_algebra_gpu_nvidia.cu)
else()
	configure_file(tensor_algebra_gpu_nvidia.cu ${PROJECT_SOURCE_DIR}/tensor_algebra_gpu_nvidia.cpp COPYONLY)
	set(TALSH_CXX_SOURCES ${TALSH_CXX_SOURCES} tensor_algebra_gpu_nvidia.cpp)
endif()

if(TALSH_GPU)
	set(TALSH_CUDA_FLAGS  ${TALSH_CUDA_FLAGS} -lineinfo -w -D_FORCE_INLINES -m64
		-Xptxas --warn-on-spills;-gencode arch=compute_${TALSH_GPU_ARCH},code=sm_${TALSH_GPU_ARCH}; -ccbin ${CMAKE_CXX_COMPILER})
	
	if(TALSH_FINE_TIMING)
		list(APPEND TALSH_CXX_COMPILE_DEFS GPU_FINE_TIMING)
		list(APPEND TALSH_Fortran_COMPILE_DEFS GPU_FINE_TIMING)
	endif()
endif()

add_library(talsh_cxx OBJECT ${TALSH_CXX_SOURCES})
target_include_directories(talsh_cxx PRIVATE ${TALSH_INCLUDES})
target_compile_definitions(talsh_cxx PRIVATE ${TALSH_CXX_COMPILE_DEFS})
target_compile_options(talsh_cxx PRIVATE ${TALSH_CXX_FLAGS})


if(TALSH_GPU)
	add_library(talsh_fort OBJECT ${TALSH_FORTRAN_SOURCES})
	target_include_directories(talsh_fort PRIVATE ${TALSH_INCLUDES})
	target_compile_definitions(talsh_fort PRIVATE ${TALSH_Fortran_COMPILE_DEFS})
	target_compile_options(talsh_fort PRIVATE ${TALSH_Fortran_FLAGS})

	set_source_files_properties(${TALSH_CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	add_library(talsh STATIC ${TALSH_CUDA_SOURCES} $<TARGET_OBJECTS:talsh_cxx> $<TARGET_OBJECTS:talsh_fort>)
	target_compile_options(talsh PRIVATE ${TALSH_CUDA_FLAGS})
	target_link_libraries(talsh PUBLIC CUDA::cublas OpenMP::OpenMP_CXX)
else()
	add_library(talsh STATIC ${TALSH_FORTRAN_SOURCES} $<TARGET_OBJECTS:talsh_cxx>)
	target_include_directories(talsh PUBLIC ${TALSH_INCLUDES})
	target_compile_definitions(talsh PRIVATE ${TALSH_Fortran_COMPILE_DEFS})
	target_compile_options(talsh PRIVATE ${TALSH_Fortran_FLAGS})
	target_link_libraries(talsh PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(talsh::talsh ALIAS talsh)

set(TALSH_HEADERS
	timer.h
	byte_packet.h
	tensor_algebra.h
	mem_manager.h
	talsh_complex.h
	talsh.h
	tensor_method.hpp
	talsh_task.hpp
	talshxx.hpp
    )

if(BUILD_TEST)
	enable_testing()
	add_executable(talsh_test main.F90 test.cpp)
	target_link_libraries(talsh_test talsh::talsh)
	target_compile_definitions(talsh_test PRIVATE ${TALSH_Fortran_COMPILE_DEFS})
	target_compile_options(talsh_test PRIVATE ${TALSH_Fortran_FLAGS})
	add_test(NAME talsh_test COMMAND talsh_test)	
endif()

set_target_properties(talsh PROPERTIES EXPORT_NAME talsh)

# Install

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/talsh)

install(TARGETS talsh
    EXPORT talsh-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


install(FILES ${TALSH_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/talsh)

#Export the targets to a script
install(EXPORT talsh-targets
    FILE
        talsh-targets.cmake
    NAMESPACE
        talsh::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/talsh-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/talsh-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/talsh-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the config, configversion files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/talsh-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/talsh-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)


export(EXPORT talsh-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/talsh-targets.cmake
    NAMESPACE talsh::)

#Register package in user's package registry
export(PACKAGE talsh)
