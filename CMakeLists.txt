cmake_minimum_required(VERSION 3.5)
project(GNN_Acceleration_Language)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
# Print the time for the compling
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

######################### Project Components ###########################
# SPADE Accelerator code-gen
option(GEN_SPADE "Build for code-gen in spade" OFF)
# Sparse transfer learning tests
option(SP_TEST "Build for code for sparse matrix ops and it's dependencies." OFF)
# GNN DSL
option(DSL "Build for code for the GNN DSL" ON)
# Enable Cuda
option(TORCH_CUDA "Build Torch CUDA code." ON)

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

# Use compressed data representation formats
option(C_COMP "Compress Sparse Matrix further. Ex - CSR -> DCSR" ON)
# TODO Somehow try to get C_COMP into the spade code generation? Or even ignore it?

option(READ_NPY "Export torch." OFF)
if (READ_NPY)
    add_compile_definitions(RNPY)
    message("Read from .npy files")
endif ()

######################### Default choices for optimizations ###########################
# This is the reordering method, no reordering 0, rabbit 1 (rabbit)
if (NOT REORD)
    set(REORD 1)
endif ()
# This is the tiling method (segmented, info based etc), default 0 (segmented)
if (NOT STILE)
    set(STILE 0)
endif ()
# This is the type of the sparse matrix used. (default is 1 to give CSR)
if (NOT SM_TYPE)
    set(SM_TYPE 1) # Default is CSR (1)
endif ()
# The preprocessing function for segmented tiling, default 2 (max parallel)
if (NOT PROC_TILE)
    set(PROC_TILE 0) # TODO SAND
endif ()
# The loop order for executing sparse matrix operations, default 2 (jj, iip, i, j, k)
if (NOT L_ORDER)
    set(L_ORDER 2)
endif ()
# The GCN Normalization method, default 1
# TODO currently tightly coupled to the sparse ops. Remove it from there.
if (NOT GCN_NORM)
    set(GCN_NORM 1) # the GCN normalization method
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
######################### Sparse transfer learning tests ###########################
if (SP_TEST)
    # Turn off MKL
    option(USE_MKL "MKL option" OFF)
    set(C_COMP OFF)  # TODO SAND

    message("Build for code for sparse matrix ops and it's dependencies.")
    add_compile_definitions(GNN_Sparse)
    if (ACC_COLD_STARTS)
        add_compile_definitions(ACC_COLD)
        message("Always have a cold starts when getting times.")
    endif ()
endif ()
######################### GNN DSL ###########################
if (DSL)
    # Turn MKL on (for the dense matrix multiplications)
    option(USE_MKL "MKL option" OFF)
    set(C_COMP ON) # Compress the sparse matrices

    message("Build for code for sparse matrix ops and it's dependencies.")
    add_compile_definitions(GNN_Sparse)

    find_package(BISON REQUIRED)
    find_package(FLEX REQUIRED)
endif ()
######################### Optimization configuration ###########################
# Compress the sparse matrix representation
if (C_COMP)
    add_compile_definitions(CCMP)
    message("Further compress the sparse matrix whenever possible. (Ex - CSR -> DCSR)")
endif ()

if (ALI_ALLOC)
    add_compile_definitions(A_ALLOC)
    message("Allocate with alligned_alloc")
endif ()

# Reorder matrix
if (REORD EQUAL 1)
    add_compile_definitions(RO_1)
    message("Do rabbit reordering.")
endif ()

# Sparse tiling options
if (STILE EQUAL 0)
    add_compile_definitions(ST_0)
    message("Tile sparse matrix using segmented tiling")
elseif (STILE EQUAL 1)
    add_compile_definitions(ST_1)
    message("Tile sparse matrix using information based tiling")
elseif (STILE EQUAL 2)
    add_compile_definitions(ST_2)
    message("Tile sparse matrix using DGL (offset re-write) based tiling")
endif ()

# Sparse matrix representation
if (SM_TYPE EQUAL 1)
    add_compile_definitions(SM_1)
    message("Sparse Matrix type is CSR")
elseif (SM_TYPE EQUAL 2)
    add_compile_definitions(SM_2)
    message("Sparse Matrix type is COO")
elseif (SM_TYPE EQUAL 3)
    add_compile_definitions(SM_3)
    message("Sparse Matrix type is DCSR ( CSR with an ID array, where only rows with NNZs would be stored. Needed for split based tiling of larger matrices.)")
endif ()

# Tile processing algorithm
if (PROC_TILE EQUAL 0)
    add_compile_definitions(PT_0)
    message("Process the tiles serially.")
elseif (PROC_TILE EQUAL 1)
    add_compile_definitions(PT_1)
    message("Process tiles parallely for each column segment.")
elseif (PROC_TILE EQUAL 2)
    add_compile_definitions(PT_2)
    message("Process tiles parallely for each row.")
endif ()

# Different types of loop orderings
if (L_ORDER EQUAL 1)
    add_compile_definitions(LO_1)
    add_compile_definitions(LO_I_P)
    message("Loop Order 1 - ii, jj, ip, j, kv")
elseif (L_ORDER EQUAL 2)
    add_compile_definitions(LO_2)
    add_compile_definitions(LO_II_P)
    message("Loop Order 2 - jj, iip, i, j, kv")
elseif (L_ORDER EQUAL 3)
    add_compile_definitions(LO_3)
    add_compile_definitions(LO_KK)
    add_compile_definitions(LO_KK_JJ_II)
    add_compile_definitions(LO_II_P)
    message("Loop Order 3 - kk, jj, iip, i, j, kv")
elseif (L_ORDER EQUAL 5)
    add_compile_definitions(LO_5)
    add_compile_definitions(LO_KK)
    add_compile_definitions(LO_JJ_KK_II)
    add_compile_definitions(LO_II_P)
    message("Loop Order 5 - jj, kk, iip, i, j, kv")
elseif (L_ORDER EQUAL 7)
    add_compile_definitions(LO_7)
    message("Order 7")
elseif (L_ORDER EQUAL 10)
    add_compile_definitions(LO_10)
    add_compile_definitions(LO_I_P)
    message("Loop Order 10 - jj, ii, ip, j, kv")
elseif (L_ORDER EQUAL 11)
    add_compile_definitions(LO_11) # This does NOT work
    add_compile_definitions(LO_K_P)
    message("Loop Order 11 - jj, ii, i, j, kvp")
elseif (L_ORDER EQUAL 12)
    add_compile_definitions(LO_12)
    add_compile_definitions(LO_KK)
    add_compile_definitions(LO_KK_P)
    message("Loop Order 12 - jj, kkp, ii, i, j, kv")
endif ()

# GNN Normalization method. # TODO Remove from the sparse section. Currently, it's tightly coupled to this.
if (GCN_NORM EQUAL 1)
    add_compile_definitions(GN_1)
    message("GCN normalization is calculated once using SDDVV.")
elseif (GCN_NORM E
QUAL 2)
    add_compile_definitions(GN_2)
    message("GCN normalization is calculated using GeMM per layer.")
endif ()

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

# Torch / Aten for the DSL
if (DSL)
#    find_package(Torch REQUIRED)
#    message("Torch cxx flags: ${TORCH_CXX_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#    message("CUDA flags: ${CMAKE_CUDA_ARCHITECTURES}")
#    message("Torch linker flags: ${TORCH_LIBRARIES}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TORCH_LIBRARIES}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltorch -ltorch_cpu -lc10")
endif ()

# if (TORCH_CUDA)
#     include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#     link_libraries("${TORCH_LIBRARIES}" cudart cusparse)
# endif ()

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
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -fopenmp=libomp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -DICC -restrict")
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
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSKYLAKE -DOMP_GT_4_5 -qopt-report=0  -march=native -xCORE-AVX512 -O3 -DICC -restrict")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSKYLAKE -DOMP_GT_4_5 -qopt-report=0  -march=native -O3 -DICC")
    if (ACC_PRE_FETCH_OFF)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-prefetch=0")
        message("Turn off prefetching.")
    endif ()
else ()
    message("Compiler: COMILER_ID is ${CMAKE_CXX_COMPILER_ID}")
endif ()

# Common compiler libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl")
# set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma")

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
#            set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -ltcmalloc_minimal")
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
        else ()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
        endif ()
    endif ()
else()
    if (GEN_SPADE OR SP_TEST)
        if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -liomp5")
        endif ()
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