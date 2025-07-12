#!/bin/bash

source ~/miniforge3/bin/activate gala
cd ../Evaluations
python Figure-19.py
source ../Environments/WiseGraph/h100/cxgnn/bin/activate
python WiseGraph.py --job F18n19 --h100
source ~/miniforge3/bin/activate gala
python Figure-19.py --job stat