# GALA Frontend Compiler
Lowers frontend GALA (Gnn Acceleration Language) code to an intermediate representation.
## Use
For convenience `input.txt` is the only input file being used. Will change later.
Simply run `make` and then `./frontend`.
## Design
First create parse tree through the grammar. Then use a post-order depth first search algorithm to search the parse tree and generate the intermediate representation.
## Environment (July 2024)
Using `Ubuntu 22.04.03` on WSL
### Compiler
Using `flex==2.6.4`, `bison==3.8` for compiler design and `g++==11.4.0` for data structures and algorithms.