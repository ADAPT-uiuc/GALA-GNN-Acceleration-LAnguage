# GNN-Acceleration-Language

## Data
Scripts necessary for downloading data can be found in `scripts/data`.

There is also a notebook to visualize two arrays of src, and dst npy files (Graph represented in COO format) to help get a visual idea of the NNZ distribution in a graph.

## Running for SPADE code generation
The scripts for SPADE are located in `scripts/spade`.
The script with the base pipeline for the code generation should be in `rmat_schedule_script/sh`

(P.S. I managed to run it with only GCC (Worked with 7.2.0 and 12.2.0, both in ICCP), BUT not sure if some dependency was already in my environment at the time.)