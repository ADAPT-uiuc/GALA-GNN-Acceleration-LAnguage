# SparseTIR Installation (July 2024)

This file simply contains installation instructions. Learn more about SparseTIR [here](https://github.com/uwsampl/SparseTIR).

## Build from source with Conda
```bash
cd gnn-benchmarks
git clone --recursive https://github.com/uwsampl/SparseTIR.git
cp sparseTIR-env.yml SparseTIR/requirements.yml
cd SparseTIR
conda env create --file requirements.yml --name sparseTIR-env
conda deactivate
conda activate sparseTIR-env
module load gcc/12.3.0
module load llvm/15.0.7
# llvm version issue
cd src/target/llvm
sed -i 's/getAlignment()/getAlign().value()/g' codegen_amdgpu.cc codegen_nvptx.cc codegen_llvm.cc
cd ../../..
mkdir build && cd build
cp ../../sparseTIRconfig.cmake config.cmake
# change below paths accordingly
export LIBRARY_PATH="/path/to/anaconda3/envs/sparseTIR-env/lib/:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/path/to/anaconda3/envs/sparseTIR-env/lib/:$LD_LIBRARY_PATH"
cmake ..
make -j$(nproc)
# install python binding
cd ../python
python setup.py install

# quick sanity check
# gcc/12.3.0 used for installation, gcc/9.5.0 used for running examples
module swap gcc/12.3.0 gcc/9.5.0-ubuntu
cd examples/spmm
python bench_spmm.py
```
