/* Definitions */
%{
    extern int yylex();
    #include <iostream>
    using namespace std;
%}

/* Only one input file */
%option noyywrap

/* Rules */
/* TODO: not sure how to deal with graph or feat as a token, they are terminals in the grammar \
    not sure if line 19 is right?, part of sample_op
    should I add quote terminals, or have that as part of the identifier terminal (e.g. "REDDIT" is all one terminal, but so is REDDIT)
*/
%%
load_dataset { cout << "LOAD "; }
dsl.get_aggregate { cout << "AGGR_INIT "; }
fn { cout << "FN_ARG "; }
dsl.fn. { cout << "DSL_FN "; }
mul_sum { cout << "MUL_SUM "; }
dsl. {cout << "DSL_DOT "; }
set_undirected { cout << "SET_UNDIRECTED "; }
set_unweighted { cout << "SET_UNWEIGHTED "; }
feature_size { cout << "FEAT_SIZE_ASSIGN "; }
label_size { cout << "LABEL_SIZE_ASSIGN "; }
coarsen { cout << "COARSEN "; }
dsl.reorder.rabbit { cout << "RABBIT_REORDER_OP "; }
dsl.sample.random  { cout << "SAMPLE_RANDOM_OP "; }
nn.ffn { cout << "FFN "; }
out= { cout << "FFN_OUT "; }
size { cout << "SIZE_FN "; }
nln.ReLU { cout << "RELU "; }
dsl.nn.loss.RMSE { cout << "RMSE_LOSS "; }
dsl.nn.optmz.ADAM { cout << "ADAM_T "; }
DSL_Dataset { cout << "DATASET "; }
NonLn { cout << "NONLN "; }
aggregate { cout << "AGGR "; }
model { cout << "MODEL_W "; }
layer { cout << "LAYER "; }
graphs { cout << "GRAPH_ATTR "; }
feats { cout << "FEAT_ATTR "; }
labels { cout << "LABEL_ATTR "; }
eval { cout << "EVAL "; }
train { cout << "TRAIN "; }
loss { cout << "LOSS "; }
optimizer { cout << "OPTIMIZER "; }
iters { cout << "ITERS "; }
validation_step { cout << "VAL_STEP "; }
relax_nln { cout << "RELAXNLN "; }
quant { cout << "QUANT "; }
sensei_op { cout << "SENSEI_OP "; }
col_tile { cout << "COLTILE "; }
null { cout << "NULL_KEY "; }

\( { cout << "LPAREN "; }
\) { cout << "RPAREN "; }
\{ { cout << "LBRACE "; }
\} { cout << "RBRACE "; }
\[ { cout << "LSQBRA "; }
\] { cout << "RSQBRA "; }
\. { cout << "DOT "; }
\, { cout << "COMMA "; }
int { cout << "INT "; }
if { cout << "IF "; }
else { cout << "ELSE "; }
true { cout << "TRUE "; }
false { cout << "FALSE "; }
new { cout << "NEW "; }
! { cout << "NOT "; }
&& { cout << "AND "; }
\|\| { cout << "OR "; }
; { cout << "SEMICOLON "; }
!= { cout << "NOTEQ "; }
== { cout << "EQ "; }
= { cout << "ASSIGN "; }
\> { cout << "GREATER "; }
\< { cout << "LESS "; }
\>= { cout << "GREATEREQ "; }
\<= { cout << "LESSEQ "; }
\+ { cout << "PLUS "; }
\- { cout << "MINUS "; }
\* { cout << "MULTIPLY "; }
\/ { cout << "DIVIDE "; }
\" { cout << "QUOTE "; }
[ \t\r\n] {}
\/\/.* { 
    cout << "COMMENT "; }
#.* { 
    cout << "COMMENT "; }
[0-9]+\.[0-9]+ {
    cout << "FLOAT ";
}
[0-9]+ {
    cout << "INTEGER ";
}
[A-Za-z_$][A-Za-z0-9_$]* {
    cout << "IDENTIFIER " ;
}
<<EOF>> { yyterminate(); }
. {}
%%

int main(int, char**){
    /* read from file instead of stdin */
    FILE *myfile = fopen("../input.txt", "r");
    if (!myfile) {
        printf("Invalid File\n");
        return -1;
    }
    yyin = myfile;
    
    while (yylex());
}