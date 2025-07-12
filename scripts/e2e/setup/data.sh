#!/bin/bash

source ~/miniforge3/bin/activate gala # Please give the correct path to activate the env if different
cd ../../Data
python get_all_datasets.py
source ../Environments/WiseGraph/h100/.venv/cxgnn2/bin/activate # replace h100 with a100 is on that machine
python get_all_datasets.py --wisegraph  # for wisegraph