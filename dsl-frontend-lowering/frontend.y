/* Definitions */
%{
#include <iostream>
#include <string.h>
#include "ir/frontendIR.h"
using namespace std;

extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);
FrontendIRNode *root;
%}

%union {
    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
}

/* TODO: add TOKENS */

%token<sval> IDENTIFIER ASSIGN LOAD;
%token<sval> LPAREN RPAREN SEMICOLON QUOTE COMMENT;
%token<sval> MODEL EVAL TRAIN LAYER LOSS OPTIMIZER ITERS VAL_STEP RMSE_LOSS ADAM;
%token<sval> RELAXNLN QUANT GRAPH_ATTR FEAT_ATTR RELU
%token<sval> RABBIT_REORDER_OP SAMPLE_RANDOM_OP
%token<sval> COLTILE AGGR
%token<sval> INTEGER;
%token<sval> FLOAT;
%token<sval> LBRACE RBRACE LSQBRA RSQBRA DOT COMMA;
%token<sval> IF ELSE DO WHILE;
%token<sval> TRUE FALSE;
%token<sval> NOT AND OR NOTEQ EQ GREATER LESS GREATEREQ LESSEQ 
%token<sval> PLUS MINUS MULTIPLY DIVIDE;
%token<sval> FFN DATASET NONLN SENSEI_OP INT NEW NULL_KEY

%type <sval> type param arg model_transform
%type <irNode> params args 
%type <irNode> dsl_prog dsl_stmnts dsl_stmnt data_stmnt 
%type <irNode> layer_defs layer_def layer_stmnts layer_stmnt
%type <irNode> model_defs model_def model_stmnts model_stmnt 
%type <irNode> model_uses model_use model_train_args model_train_arg
%type <irNode> graph_ds.graph graph_ds.feat
/* TODO:
*/
%%      
dsl_prog : { $$ = NULL; }
    | dsl_stmnts
    {
        $$ = new FrontendIRNode("dsl_prog");
        $$->addChild($1);
        root = $$;
    }
    | dsl_stmnts layer_defs model_defs model_stmnts model_uses dsl_stmnts
    {
        $$ = new FrontendIRNode("dsl_prog");
        $$->addChild($1);
        $$->addChild($2);
        $$->addChild($3);
        $$->addChild($4);
        if ($5) $$->addChild($5);
        if ($6) $$->addChild($6);
        root = $$;
    }
;
dsl_stmnts : { $$ = NULL; }
    | dsl_stmnts dsl_stmnt
    {
        $$ = new FrontendIRNode("dsl_stmnts");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
dsl_stmnt : IDENTIFIER ASSIGN data_stmnt
    {
        $$ = new FrontendIRNode("data_stmnt");
        $$->addParam($1);
        $$->addChild($3);
        free($1); // using free instead of delete b/c don't want to delete children nodes
    }
    | COMMENT 
    { 
        $$ = new FrontendIRNode("comment"); 
        $$->addParam($1); 
        free($1); 
    }
    | IDENTIFIER ASSIGN graph_ds.graph SEMICOLON
    {
        $$ = new FrontendIRNode("graph_access");
        $$->addParam($1);
        $$->addChild($3);
        free($1);
    }
    | IDENTIFIER ASSIGN graph_ds.feat SEMICOLON
    {
        $$ = new FrontendIRNode("feat_access");
        $$->addParam($1);
        $$->addChild($3);
        free($1);
    }
    /*
    | IDENTIFIER ASSIGN gnn_op
    | IDENTIFIER ASSIGN fuse_ops
    | IDENTIFIER ASSIGN new_op
    | IDENTIFIER ASSIGN var
    */
;
data_stmnt : LOAD LPAREN QUOTE IDENTIFIER QUOTE RPAREN SEMICOLON
    {
        $$ = new FrontendIRNode("load");
        $$->addParam($4);
        free($4);
    }
    /*
    | graph.graph_data_transform
    | feature.dense_data_transform
    | graph_ds.data_transform
    */
;
graph_ds.graph : IDENTIFIER DOT GRAPH_ATTR
{
    $$ = new FrontendIRNode("graph_attr");
    $$->addParam($1);
    free($1);
}
;
graph_ds.feat : IDENTIFIER DOT FEAT_ATTR
{
    $$ = new FrontendIRNode("feat_attr");
    $$->addParam($1);
    free($1);
}
layer_defs : layer_def 
    { 
        $$ = new FrontendIRNode("layer_defs");
        $$->addChild($1);
    }
    | layer_defs layer_def 
    {
        $$ = new FrontendIRNode("layer_defs");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
// identifier is layer_spec
layer_def : IDENTIFIER ASSIGN LAYER LPAREN params RPAREN LBRACE dsl_stmnts RBRACE
    {
        $$ = new FrontendIRNode("layer_def");
        $$->addParam($1);
        if ($5) $$->addChild($5); // L1 = layer(args), in this case args is a parameter, might change
        if ($8) $$->addChild($8); // dsl_stmnts are optional for now, necessary later
        free($1);
    }
;
layer_stmnts : layer_stmnt 
    { 
        $$ = new FrontendIRNode("layer_stmnts");
        $$->addChild($1);
    }
    | layer_stmnt layer_stmnt 
    {
        $$ = new FrontendIRNode("layer_stmnts");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
            // first identifier is layer_var, second is layer_spec
layer_stmnt : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON
    {
        $$ = new FrontendIRNode("layer_stmnt");
        $$->addParam($1);
        $$->addParam($3);
        if ($5) $$->addChild($5); // args
        
        free($1);
    }
;
model_defs : model_def
    {
        $$ = new FrontendIRNode("model_defs");
        $$->addChild($1);
    }
    | model_defs model_def
    {
        $$ = new FrontendIRNode("model_defs");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
            // IDENTIFIER is model_spec
model_def : IDENTIFIER ASSIGN MODEL LPAREN params RPAREN LBRACE layer_stmnts RBRACE
    {
        $$ = new FrontendIRNode("model_def");
        $$->addParam($1);
        if ($5) $$->addChild($5); 
        if ($8) $$->addChild($8); 
    }
;
model_stmnts : model_stmnt 
    { 
        $$ = new FrontendIRNode("model_stmnts");
        $$->addChild($1);
    }
    | model_stmnts model_stmnt
    {
        $$ = new FrontendIRNode("model_stmnts");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
// first identifier is model_var, second is model_spec
model_stmnt : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON
    {
        $$ = new FrontendIRNode("model_stmnt");
        $$->addParam($1);
        $$->addParam($3);
        if ($5) $$->addChild($5); // args
        
        free($1);
    }
;
model_uses : model_use 
    {
        $$ = new FrontendIRNode("model_uses");
        $$->addChild($1);
    }
    | model_uses model_use
    {
        $$ = new FrontendIRNode("model_uses");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
 //example doesn't use graph_ds parameter but grammar needs?
  // example has variable assigned but grammar doesn't have?

model_use : IDENTIFIER DOT TRAIN LPAREN model_train_args RPAREN SEMICOLON
    {
        $$ = new FrontendIRNode("model_use_train");
        $$->addParam($1);
        if ($5) $$->addChild($5);
        free($1);
    }
    /* no graph_ds param in eval because already passed in the model creation
    going to have the eval method return a variable (maybe an array of embeddings) instead of just being void
    */
    | IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN RPAREN SEMICOLON
    {
        $$ = new FrontendIRNode("model_use_eval");
        $$->addParam($1);
        $$->addParam($3);
    }
    | IDENTIFIER DOT model_transform SEMICOLON
    {
        $$ = new FrontendIRNode("model_use_transform");
        $$->addParam($3);
    }
;

/* explicity assigning variables is necessary for now
 based on grammar, need one or more model_train_arg 
 it is not specified which one though, so any one model_train_arg will make code pass without syntax error
*/
model_train_args : model_train_arg
    { 
        $$ = new FrontendIRNode("model_train_args");
        $$->addChild($1);
    }
    | model_train_args model_train_arg
    {
        $$ = new FrontendIRNode("model_train_args");
        if ($1){
            for (auto child : $1->children){
                $$->addChild(child);
            }
            free($1);
        }
        $$->addChild($2);
    }
;
model_train_arg : ITERS ASSIGN INTEGER
    {
        $$ = new FrontendIRNode("iters");
        $$->addParam($3);
    }
    | ITERS ASSIGN INTEGER COMMA
    {
        $$ = new FrontendIRNode("iters");
        $$->addParam($3);
    }
    | LOSS ASSIGN RMSE_LOSS
    {
        $$ = new FrontendIRNode("loss");
        $$->addParam("rmse");
    }
    | LOSS ASSIGN RMSE_LOSS COMMA
    {
        $$ = new FrontendIRNode("loss");
        $$->addParam("rmse");

    }
    | OPTIMIZER ASSIGN ADAM
    {
        $$ = new FrontendIRNode("optimizer");
        $$->addParam("adam");
    }
    | OPTIMIZER ASSIGN ADAM COMMA
    {
        $$ = new FrontendIRNode("optimizer");
        $$->addParam("adam");
    }
    | VAL_STEP ASSIGN INTEGER
    {
        $$ = new FrontendIRNode("validation_step");
        $$->addParam($3);
    }
    | VAL_STEP ASSIGN INTEGER COMMA
    {
        $$ = new FrontendIRNode("validation_step");
        $$->addParam($3);
    }
;
/*
is this variable assigned (probably not)?
*/
model_transform : RELAXNLN LPAREN RPAREN
    {
        $$ = (char*) malloc(strlen("relax_nln")+1);
        sprintf($$, "relax_nln");
    }
    | QUANT LPAREN RPAREN 
    {
        $$ = (char*) malloc(strlen("quant")+1);
        sprintf($$, "quant");
    }
    | SENSEI_OP LPAREN RPAREN
    {
        $$ = (char*) malloc(strlen("sensei_op")+1);
        sprintf($$, "sensei_op");
    }
;
params : { $$ = NULL; } 
    | params param 
    {
        $$ = new FrontendIRNode("params");
        if ($1){
            for (string arg : $1->params)
                $$->addParam(arg);
            delete $1;
        }
        
        $$->addParam($2);
        free($2);
    }
;
param : type IDENTIFIER COMMA 
    {
        $$ = (char*) malloc(strlen($1) + strlen($2) + 2);
        sprintf($$, "%s %s", $1, $2);
        free($1);
        free($2);
    }
    | type IDENTIFIER
    {
        $$ = (char*) malloc(strlen($1) + strlen($2) + 2);
        sprintf($$, "%s %s", $1, $2);
        free($1);
        free($2);
    }
;
args : { $$ = NULL; } 
    | args arg 
    {
        $$ = new FrontendIRNode("args");
        if ($1){
            for (string arg : $1->params)
                $$->addParam(arg);
            delete $1;
        }
        
        $$->addParam($2);
        free($2);
    }
;
arg : IDENTIFIER COMMA 
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | IDENTIFIER
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | INTEGER COMMA
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | INTEGER
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | FLOAT COMMA
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | FLOAT 
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | NULL_KEY COMMA
    {
        $$ = (char*) malloc(strlen($1) + 2);
        sprintf($$, "%s", $1);
        free($1);
    }
    | NULL_KEY
    {
        $$ = strdup("NULL");
    }
    | RELU COMMA
    {
        $$ = strdup("dsl.nln.ReLU");
    }
    | RELU
    {
        $$ = strdup("dsl.nln.ReLU");
    }
;
type : DATASET { $$ = strdup("DSL_Dataset"); } 
    | NONLN { $$ = strdup("NonLn"); }
    | INT { $$ = strdup("int"); }
;

%%

/* C Code */
int main(int argc, char** argv) {
    /* read from file instead of stdin */
    if (argc < 2){
        cout << "no filename provided";
        return 1;
    }
    const char* fileName= argv[1];
    FILE *myfile = fopen(fileName, "r");
    if (!myfile) {
        cout << "Invalid File" << endl;
        return -1;
    }
    yyin = myfile;
    yyparse();
    cout << "------- generating parse tree -----\n";

    int directChildren = FrontendIRNode::countDirectChildren(root);
    vector<bool> flag(directChildren, true); 
    FrontendIRNode::printParseTree(root, std::cout, flag);
    /* FrontendIRNode::generateIR(root, cout); */
    delete root;

    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}