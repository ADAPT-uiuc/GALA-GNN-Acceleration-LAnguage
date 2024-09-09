/* Definitions */
%{
    extern int yylex();
    #include "ir/frontendIR.h"
    #include "frontend.tab.h"
    using namespace std;
%}

/* Only one input file */
%option noyywrap

/* Rules */
/* TODO: not sure how to deal with graph or feat as a token, they are terminals in the grammar \
    not sure if line 19 is right?, part of sample_op
    add specific terminals for model_var, layer_var, model_spec, etc..
    confused about model_spec. 
*/
%%
dsl.load { return LOAD; }
dsl.reorder.rabbit { return RABBIT_REORDER_OP; }
dsl.sample.random  { return SAMPLE_RANDOM_OP; }
dsl.nn.ffn { return FFN; }
dsl.nln.ReLU { return RELU; }
dsl.nn.loss.RMSE { return RMSE_LOSS; }
dsl.nn.optmz.ADAM { return ADAM; }
DSL_Dataset { return DATASET; }
NonLn { return NONLN; }
aggregate { return AGGR; }
model { return MODEL; }
layer { return LAYER; }
graph { return GRAPH_ATTR; }
feat { return FEAT_ATTR; }
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
\/\/.* { 
    yylval.sval = strdup(yytext);
    return COMMENT; 
}
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