# GNN Benchmarks

Perform benchmarks on various GNN frameworks and models.\
Customize number of layers, feature sizes, and embedding sizes.\
\
Has `DGL`, `PyG`, and `Graphiler`. More frameworks/compilers will be added soon.
## Install (DGL & PyG) with Conda
Simply clone, create a conda environment, and run `main.py`

```bash
  git clone "https://github.com/nikhiljayakumar/gnn-benchmarks"
  cd gnn-benchmarks
  conda env create --file benchmarks-env.yml --name benchmarks-env
  conda deactivate
  conda activate benchmarks-env
  python main.py tests.txt
```
## Graphiler
Please see [README_graphiler.md](README_graphiler.md) for installation instructions. Once installed, activate environment and run `main.py`
Graphiler tests MUST be seperate from other tests. This is because a different environment is needed and thus cannot be ran simultaneously.
```bash
cd gnn-benchmarks
conda activate graphiler-env
export GRAPHILER=$(pwd)
export LD_LIBRARY_PATH="/srv/home/<netid>/anaconda3/envs/graphiler-env/lib/:$LD_LIBRARY_PATH"
python main.py graphilerTests.txt
```
## SparseTIR
Please see [README_sparseTIR.md](README_sparseTIR.md) for installation instructions. Once installed, activate environment and run `main.py`
```bash
cd gnn-benchmarks
conda activate sparseTIR-env
# change below paths accordingly
export LIBRARY_PATH="/path/to/anaconda3/envs/sparseTIR-env/lib/:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/path/to/anaconda3/envs/sparseTIR-env/lib/:$LD_LIBRARY_PATH"
module load gcc/9.5.0
# current fix, working on building proper GNN models
cd SparseTIR/examples/spmm
python bench_spmm.py
```
## Run Benchmark
Benchmark(s) should be made into a single file in the same directory as `main.py` and entered while running script
`python main.py tests.txt`
A benchmark can be entered through a string of inputs in the following format.
```bash
'framework' 'model' 'hidden-layer-sizes' 'dataset' 'epochs' 'heads'
```
`heads` is only necessary for Graph Attention Networks\
\
For instance, running 200 epochs on a Graph Attention Network with DGL on the Cora Dataset with a hidden layer size of 16 and 8, 1 attention heads for the first and second layer respectively, enter the following:
```bash
dgl gat [16] Cora 200 [8, 1]
```
See `tests.txt` for more examples
