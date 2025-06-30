pip install torch==2.1.0
yes | pip uninstall triton
pip install "numpy<2"

cd ../triton/python
git checkout release/2.1.x
python setup.py build -j32 develop
cd ../../CxGNN-DL
python setup.py build -j32 develop
cd ../CxGNN-Compute
python setup.py build -j32 develop

pip install  dgl==1.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install -r requirements.txt
pip install torch_geometric
pip install ogb pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html