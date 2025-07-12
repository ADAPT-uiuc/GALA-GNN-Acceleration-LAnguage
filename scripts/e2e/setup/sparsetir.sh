#!/bin/bash

cd ../../Environments/SparseTIR
git clone --recursive https://github.com/uwsampl/SparseTIR.git
# For H100
cp h100_sparseTIR_env.yml SparseTIR/requirements.yml
# Else
cp a100_sparseTIR_env.yml SparseTIR/requirements.yml
cd SparseTIR
conda deactivate
conda env create --file requirements.yml --name stir-gala-ae
source ~/miniforge3/bin/activate stir-gala-ae
conda install -c conda-forge gcc=12.3.0 gxx=12.3.0 llvm=15.0.7 clang=15.0.7 llvmdev=15.0.7
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export LLVM_CONFIG=$CONDA_PREFIX/bin/llvm-config
# llvm version issue
cd src/target/llvm
sed -i 's/getAlignment()/getAlign().value()/g' codegen_amdgpu.cc codegen_nvptx.cc codegen_llvm.cc
cd ../../..
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
mkdir -p build && cd build
cp ../../sparseTIRconfig.cmake config.cmake
# Below Linen Only for H100
ln -s $CONDA_PREFIX/lib/librhash.so.1 $CONDA_PREFIX/lib/librhash.so.0
cmake ..     -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/gcc     -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/g++     -DLLVM_CONFIG=$CONDA_PREFIX/bin/llvm-config     -DUSE_CUDA=ON     -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX     -DCUDA_CUDA_LIBRARY=$CONDA_PREFIX/lib/stubs/libcuda.so   -DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib/stubs/
make -j$(nproc)
# install python binding
cd ../python
python setup.py install
# gcc/12.3.0 used for installation, gcc/9.5.0 used for running examples
conda install -c conda-forge gcc=9.5.0 gxx=9.5.0
# quick sanity check
cd ../examples/spmm
python bench_spmm.py