/* Definitions */
%{
    extern int yylex();
    #include "ir/data.h"
    #include "ir/compute.h"
    #include "ir/frontendIR.h"
    #include "frontend.tab.h"
    using namespace std;
    
    /* if there ever ends up being a comment token, this is how you do it
     \/\/.* { 
        yylval.sval = strdup(yytext);
        return COMMENT; 
    } */
%}

/* Only one input file */
%option noyywrap

/* Rules */
/* TODO: not sure how to deal with graph or feat as a token, they are terminals in the grammar \
    not sure if line 19 is right?, part of sample_op
    add specific terminals for model_var, layer_var, model_spec, etc..
    confused about model_spec. 
    chatgpt says to use /b(True|False)/b for bool token (/b does boundaries)
       to avoid stuff like TTrue or Falsee
*/
%%
load_dataset { return LOAD; }
dsl.get_aggregate { return AGGR_INIT; }
fn { return FN_ARG; }
dsl.fn. { return DSL_FN; }
mul_sum { return MUL_SUM; }
dsl. {return DSL_DOT; }
dsl.reorder.rabbit { return RABBIT_REORDER_OP; }
dsl.sample.random  { return SAMPLE_RANDOM_OP; }
nn.ffn { return FFN; }
out= { return FFN_OUT; }
size { return SIZE_FN; }
nln.ReLU { return RELU; }
dsl.nn.loss.RMSE { return RMSE_LOSS; }
dsl.nn.optmz.ADAM { return ADAM_T; }
DSL_Dataset { return DATASET; }
NonLn { return NONLN; }
aggregate { return AGGR; }
model { return MODEL_W; }
layer { return LAYER; }
graphs { return GRAPH_ATTR; }
feats { return FEAT_ATTR; }
labels { return LABEL_ATTR; }
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
true { return TRUE; }
false { return FALSE; }
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
[0-9]+\.[0-9]+ {
    yylval.sval = strdup(yytext);
    return FLOAT; 
}
[0-9]+ {
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