/* Definitions */
%{
extern int yylex();
#include "../ir/data.h"
#include "../ir/compute.h"
#include "../ir/frontend_metadata.h"
//#include "frontend.tab.h"
#include "parser.hpp"
using namespace std;

/* comment token (if needed)
    \/\/.* {
    yylval.sval = strdup(yytext);
    return COMMENT;
} */
%}

/* Only one input file */
%option noyywrap

/* Rules */
%%
load_dataset { return LOAD; }
dsl.get_aggregate { return AGGR_INIT; }
dsl.get_edge_aggregate { return EDGE_AGGR_INIT; }
fn.sum { return SUM; }
fn.mul_sum { return MUL_SUM; }
fn.mul_mean { return MUL_MEAN; }
set_undirected { return SET_UNDIRECTED; }
set_unweighted { return SET_UNWEIGHTED; }
feature_size { return FEAT_SIZE_ASSIGN; }
label_size { return LABEL_SIZE_ASSIGN; }
coarsen { return COARSEN; }
operator_reordering { return OP_REORD; }
sparse_rewrites { return SPARSE_REWRITES; }
training_subgraph { return TRAIN_SUBGRAPH; }
train_code_motion { return TRAIN_CODE_MOTION; }
dsl.reorder.rabbit { return RABBIT_REORDER_OP; }
dsl.sample.random  { return SAMPLE_RANDOM_OP; }
nn.ffn { return FFN; }
nn.scalar { return SCALAR_INIT; }
fn.softmax { return SOFTMAX; }
nn.init_weight { return INIT_WEIGHT; }
fn.pow { return POW; }
fn { return FN_ARG; }
dsl. {return DSL_DOT; }
src { return SRC_ATTR; }
dst { return DST_ATTR; }
out= { return FFN_OUT; }
size { return SIZE_FN; }
non_ln.ReLU { return RELU; }
non_ln.LeakyReLU { return LEAKY_RELU; }
dsl.nn.loss.RMSE { return RMSE_LOSS; }
dsl.nn.optmz.ADAM { return ADAM_T; }
DSL_Dataset { return DATASET; }
NonLn { return NONLN; }
aggregate { return AGGR; }
model { return MODEL_W; }
layer { return LAYER; }
graphs { return GRAPH_ATTR; }
node { return NODE_ATTR; }
feats { return FEAT_ATTR; }
edges { return EDGE_ATTR; }
vals { return VAL_ATTR; }
labels { return LABEL_ATTR; }
degrees { return DEGREE_ATTR; }
eval { return EVAL; }
train { return TRAIN; }
loss { return LOSS; }
optimizer { return OPTIMIZER; }
iters { return ITERS; }
validation_step { return VAL_STEP; }
relax_nln { return RELAXNLN; }
quant { return QUANT; }
sensei_op { return SENSEI_OP; }
col_tile { return COLTILE; }
null { return NULL_KEY; }

\( { return LPAREN; }
\) { return RPAREN; }
\{ { return LBRACE; }
\} { return RBRACE; }
\[ { return LSQBRA; }
\] { return RSQBRA; }
\. { return DOT; }
\, { return COMMA; }
int { return INT; }
if { return IF; }
else { return ELSE; }
true { return TR; }
false { return FA; }
new { return NEW; }
! { return NOT; }
&& { return AND; }
\|\| { return OR; }
; { return SEMICOLON; }
!= { return NOTEQ; }
== { return EQ; }
= { return ASSIGN; }
\> { return GREATER; }
\< { return LESS; }
\>= { return GREATEREQ; }
\<= { return LESSEQ; }
\+ { return PLUS; }
\- { return MINUS; }
\* { return MULTIPLY; }
\/ { return DIVIDE; }
\" { return QUOTE; }
[ \t\r\n] {}
\/\/.* {}
#.* {}
-?[0-9]+\.[0-9]+ {
    yylval.sval = strdup(yytext);
    return FLOAT; 
}
-?[0-9]+ {
    yylval.sval = strdup(yytext);
    return INTEGER; 
}
[A-Za-z_$][A-Za-z0-9_$]* {
    yylval.sval = strdup(yytext);
    return IDENTIFIER;
}
<<EOF>> { yyterminate(); }
. {}
%%