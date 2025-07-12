# GNN-Acceleration-Language

## Directory structure

```commandline
project-root/
├── codegen/ # output path for GALA code generation
├── Data/ # path where datasets are stores
├── scripts/ # various scripts for setup and reproduce results
│   └── Data # download datasets
│       └── get_all_datasets.py # Main script that downloads datasets
│       └── other auxiliary scripts
│   └── e2e # end-2-end bash scripts to generate results and setup
│       └── setup # setup scripts
│       └── scripts for figures and tables (f<number>.sh or t<number>.sh)
│       └── simple-test.sh # script for a simple test of GALA
│   └── Environments # scripts/code necessary for the setup of baselines
│   └── Evaluations # scripts to reproduce Figures and Tables 
│   # NOTE: These require multiple runs (to get results, then create figure) with different input parameters
│       └── Figure-<number>.py # script to generate results and figure
│       └── Table-<number>.py # script to generate results and table
│       └── WiseGraph.py # script to specifically generaet WiseGraph results
├── src/ # Source code for GALA
├── tests/ # Baseline test code and GALA DSL and Compiler for testing
│   └── Baselines # Code for baselines
│   └── GALA-DSL # GALA front-end code for testing
│       └── ablations
│           └── input-optimize # DSL for input optimizer tests
│           └── memory-consumption  # DSL for memory consumption tests
│           └── sampling # DSL for sampling tests
│           └── scalability # DSL for scalability tests
│           └── speedups # DSL for other speedup breakdown tests
│       └── gat # DSL for the GAT model
│       └── gcn # DSL for the GCN model
│       └── gin # DSL for the GIN model
│       └── sage # DSL for the SAGE model
│   └── *.cpp # GALA Compiler executables
├── utils/ # Utils used in GALA
└── README.md # readme
```
## Prerequisites
* Cuda Toolkit 
* Conda
* g++ >= 11.4
* Cmake >= 3.25

### Installation (Linux/Ubuntu)
```angular2html
cd scripts/e2e/setup/
./gala.sh
./data.sh # the minimal necessary to run GALA (there are other scripts for baselines in the same directory)
```

## Quick test for GALA
```angular2html
cd scripts/e2e/
./simple-test.sh
# OR
cd build
tests/gala_inference <path to GALA DSL code> <output path> # run the codegen 
cd <output path>
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<path to libtorch> ..
make -j6
./gala_model
```

[//]: # (Others models are GIN and GAT.)

[//]: # (This should generate the final executable code in the `test-codegen` folder.)

[//]: # ()
[//]: # (The test files are in the `tests` folder, with the name `gala_<model>_IR.cpp`.)

[//]: # (Currently, the manual IR in uncommented, and the IR generation from the front-end language is commented &#40;needs to fix some bugs&#41;.  )

## Data
Scripts necessary for downloading data can be found in `scripts/Data`.

[//]: # (There is also a notebook to visualize two arrays of src, and dst npy files &#40;Graph represented in COO format&#41; to help get a visual idea of the NNZ distribution in a graph.)