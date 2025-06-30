# CxGNN-DL

Data loading component for CxGNN, including different graph sampling methods such as neighbor sampling, cluster sampling, and layer-wise sampling. Able to produce DGL/PyG-friendly graph samples.

## install
```
source /opt/spack/share/spack/setup-env.sh
spack load /5juudln # cuda11.3
spack load /7zlelqx # cudnn8.2.4
python setup.py build -j16 develop --user
```

## run
```
python example/run_cxg_loader.py type=cxg dataset=arxiv
python example/run_cxg_loader.py type=dgl dataset=papers100M
```
