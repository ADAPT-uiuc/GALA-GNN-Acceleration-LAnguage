project(GNN_Acceleration_Language)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

# Print the time for the compling
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

######################### General Build options ###########################
# Build with debug info or not
option(NO_DEBUG "Give random base addresses" ON)
if (NO_DEBUG)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
else ()
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Use aligned alloc instead of the standard alloc
option(ALI_ALLOC "Allocate with alligned_alloc" ON)

# Configure the project to build for either SPADE Code-gen OR the GNN DSL
option(GEN_SPADE "Build for code-gen in spade + sparse only without the DSL and it's dependencies." ON)

# Use compressed data representation formats
option(C_COMP "Compress Sparse Matrix further. Ex - CSR -> DCSR" ON)
# TODO Somehow try to get C_COMP into the spade code generation? Or even ignore it?

option(READ_NPY "Export torch." ON)
if (READ_NPY)
    add_compile_definitions(RNPY)
    message("Read from .npy files")
endif ()

# Reorder matrix
if (NOT REORD)
    set(REORD 1)
endif ()
if (REORD EQUAL 1)
    add_compile_definitions(RO_1)
    message("Do rabbit reordering.")
endif ()

######################### SPADE Accelerator code-gen ###########################
if (GEN_SPADE)
    option(ACC_RANDOM_BASE "Give random base addresses" ON)
    option(ACC_COLD_STARTS "Always have a cold starts when getting times" OFF)
    option(ACC_PRE_FETCH_OFF "Disable prefetching" OFF) # TODO This could be removed.

    # Turn off MKL
    option(USE_MKL "MKL option" OFF)
    set(C_COMP OFF)

    message("Build for code-gen in spade + sparse only without the DSL and it's dependencies.")
    add_compile_definitions(GNN_Sparse)
    add_compile_definitions(SPADE)
    if (ACC_RANDOM_BASE)
        add_compile_definitions(ACC_RB)
        message("Give random base addresses.")
    endif ()
    if (ACC_COLD_STARTS)
        add_compile_definitions(ACC_COLD)
        message("Always have a cold starts when getting times.")
    endif ()
endif ()

######################### GNN DSL ###########################
# TODO Fill up this area

######################### Compiler Setup ###########################
message("Using compiler: ${CMAKE_CXX_COMPILER_ID}")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

# Check include files
INCLUDE(CheckIncludeFiles)
CHECK_INCLUDE_FILES(malloc.h HAVE_MALLOC_H)
CHECK_INCLUDE_FILES(sys/mman.h HAVE_MMAN_H)
CHECK_INCLUDE_FILES(sys/sysinfo.h HAVE_SYSINFO_H)
CHECK_INCLUDE_FILES(sys/types.h HAVE_TYPES_H)
CHECK_INCLUDE_FILES(unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES(sys/syscall.h HAVE_SYSCALL_H)

# Always use -O3
add_compile_options(-O3)
option(CHECK_MEM_LEAK "MEM LEAK option" OFF) # TODO Turn on??
message(STATUS "Compiling for ${ARCH}")


if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    message("Compiler: Identifies as Clang")
    # Set settings based on the compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -fopenmp=libomp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    message("Compiler: Identifies as Intel")
    # Set settings based on the compiler
    add_compile_definitions(ICC_CMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -DICC -restrict")
    if (ACC_PRE_FETCH_OFF)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-prefetch=0")
        message("Turn off prefetching.")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    message("Compiler: Identifies as GNU")
    # Set settings based on the compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -DICC -pthread")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
    message("Compiler: Identifies as Intel (LLVM)")
    # Set settings based on the compiler
    add_compile_definitions(ICC_CMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSKYLAKE -DOMP_GT_4_5 -qopt-report=0  -march=native -xCORE-AVX512 -O3 -DICC -restrict")
    if (ACC_PRE_FETCH_OFF)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-prefetch=0")
        message("Turn off prefetching.")
    endif ()
else ()
    message("Compiler: COMILER_ID is ${CMAKE_CXX_COMPILER_ID}")
endif ()

# Common compiler libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl")
set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma")

# Set compiler flags based on the use or not mkl
if (USE_MKL)
    if (DEFINED $ENV{MKLROOT})
        message(FATAL_ERROR "USE_MKL is on but MKLROOT is not set")
    else ()
        message(STATUS "MKLROOT set to $ENV{MKLROOT}")


        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{MKLROOT}lib/intel64 -lmkl_core")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -I$ENV{MKLROOT}include")

        # Set settings based on what MKL lp library to use
        if (MKL_ILP)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_ilp64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_lp64")
        endif()

        # Set settings based on the compiler
        if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_gnu_thread -lgomp")
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
            set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -ltcmalloc_minimal")
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
        else ()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
        endif ()
    endif ()
else()
    if (GEN_SPADE)
        if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -liomp5")
        endif ()
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp -lopenblas")
    endif ()
endif ()

if (CHECK_MEM_LEAK)
    if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
    else ()
        message(FATAL_ERROR "Memory leak check not available")
    endif ()
endif ()

# Output the compiler flags
message("CMAKE_CXX_FLAGS - ${CMAKE_CXX_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS - ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_CXX_STANDARD_LIBRARIES - ${CMAKE_CXX_STANDARD_LIBRARIES}")

add_subdirectory(tests)