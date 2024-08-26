# Graphiler Installation (May 2024)

This file simply contains installation instructions. Learn more about graphiler [here](https://github.com/xiezhq-hermann/graphiler/tree/main).

## Docker
See [the original repository](https://github.com/xiezhq-hermann/graphiler/tree/main) to pull or directly build a docker image of graphiler.

## Build from source with Conda
```bash
cd gnn-benchmarks
git clone https://github.com/xiezhq-hermann/graphiler.git
cp graphiler-env.yml graphiler/requirements.yml
conda env create --file requirements.yml --name graphiler-env
conda deactivate
conda activate graphiler-env
mkdir build && cd build
module load gcc/9.5.0-ubuntu
module load cmake/3.25.1
export PATH=/software/gcc-9.5.0-ubuntu/gcc-9.5.0-ubuntu/bin:$PATH
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
mkdir -p ~/.dgl
mv libgraphiler.so ~/.dgl/
cd ..
python setup.py install
# path used in scripts, change accordingly
export GRAPHILER=$(pwd)
export LD_LIBRARY_PATH="/path/to/anaconda3/envs/graphiler-env/lib/:$LD_LIBRARY_PATH"

# quick sanity check
python $GRAPHILER/examples/GAT/GAT.py pubmed 500
```
