# GALA Frontend Compiler
Lowers frontend GALA (Gnn Acceleration Language) code to an intermediate representation.
## Use
For convenience `input.txt` is the only input file being used. Will change later.
Simply run `make` and then `./frontend {filename}`.
## Design
First create parse tree through the grammar. Then use a post-order depth first search algorithm to search the parse tree and generate the intermediate representation.
## Environment (July 2024)
Using `Ubuntu 22.04.03` on WSL
### Compiler
Using `flex==2.6.4`, `bison==3.8` for compiler design and `g++==11.4.0` for data structures and algorithms.
### Current Error (9/9/24)
Parser is supposed to work for an input seen in `buggyInput.txt` where after a model is evaluated other data_stmnts (child of dsl_stmnts) can be called. But it doesn't parse. However, it only works for `workingInput.txt` where a comment is placed before the data_stmnts. 