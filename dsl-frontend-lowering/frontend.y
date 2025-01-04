%{
#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include "ir/data.h"
#include "ir/compute.h"
#include "ir/frontendIR.h"
using namespace std;

extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);
vector<CIRNode*> program;
vector<RelationEdge*> dependencies;
vector<RelationEdge*> associations;
vector<TransformEdge*> transforms;
map<string, DataNode*> dataNodeMap;
%}

%union {
    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
    ForwardNode* forwardNode;
    TrainingLoopNode* trainingLoopNode;
}

%token<sval> IDENTIFIER ASSIGN LOAD;
%token<sval> LPAREN RPAREN SEMICOLON QUOTE COMMENT
%token<sval> MODEL_W EVAL TRAIN LAYER LOSS OPTIMIZER ITERS VAL_STEP RMSE_LOSS ADAM_T
%token<sval> AGGR_INIT FN_ARG MUL_SUM DSL_FN DSL_DOT FFN_OUT SIZE_FN 
%token<sval> RELAXNLN QUANT GRAPH_ATTR FEAT_ATTR RELU LABEL_ATTR
%token<sval> RABBIT_REORDER_OP SAMPLE_RANDOM_OP
%token<sval> COLTILE AGGR
%token<sval> INTEGER FLOAT
%token<sval> LBRACE RBRACE LSQBRA RSQBRA DOT COMMA;
%token<sval> IF ELSE DO WHILE;
%token<sval> TRUE FALSE;
%token<sval> NOT AND OR NOTEQ EQ GREATER LESS GREATEREQ LESSEQ 
%token<sval> PLUS MINUS MULTIPLY DIVIDE;
%token<sval> FFN DATASET NONLN SENSEI_OP INT NEW NULL_KEY
/* %token<sval> IDENTIFIER ASSIGN LOAD;
%token<sval> LPAREN RPAREN SEMICOLON QUOTE COMMENT;
%token<sval> INTEGER FLOAT;
%token<sval> MODEL_W EVAL TRAIN LAYER GRAPH_ATTR
%token<sval> ITERS LOSS OPTIMIZER VAL_STEP TEST_STEP
%token<sval> LBRACE RBRACE LSQBRA RSQBRA DOT COMMA;
%token<sval> IF ELSE DO WHILE;
%token<sval> TRUE FALSE;
%token<sval> NOT AND OR NOTEQ EQ GREATER LESS GREATEREQ LESSEQ 
%token<sval> PLUS MINUS MULTIPLY DIVIDE; */

%type <sval> arg train_arg string data_var function update_op gnn_op
%type <forwardNode> load_dataset function_init statement
%type <trainingLoopNode> algorithm layers layer_def statements
%type <irNode> program schedule args train_args
%type <irNode> layer_inits layer_init
%type <irNode> model model_def model_init model_uses model_use


%%
program : load_dataset algorithm // TODO: add schedule
    {
        program.push_back($1);
        program.push_back($2);
    }
;
load_dataset : IDENTIFIER ASSIGN LOAD LPAREN string RPAREN SEMICOLON 
    {
        $$ = new ForwardNode(POINTWISE, LOAD_OP);
        $$->addParam($5); 
        // Graph
        DataInfo* graphInfo = new DataInfo(CSR_STYPE, false, false);
        DataLevel* rootGraphLevel = new DataLevel(graphInfo, true);
        DataNode* graphData = new DataNode("Graph", INT32, INT32, F32, rootGraphLevel);
        // Feat
        DataInfo* featInfo = new DataInfo(RM_DTYPE);
        featInfo->setDims(232965, 605); // TODO: Case statement for string keyword to input how many nodes and feature size
        DataLevel* rootFeatLevel = new DataLevel(featInfo, true);
        DataNode* featData = new DataNode("Feat", INT32, INT32, F32, rootFeatLevel);

        dataNodeMap["Graph"] = graphData;
        dataNodeMap["Feat"] = featData;


        // Relation (association) between graph and features
        RelationEdge* graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(graphFeatAssociation);

        $$->addOutputData(featData);
        $$->addOutputData(graphData);
        
        // graph transformation -!-!- automatically happen, or should put somewhere else?)
        DataLevel* originalRootGraphLevel = graphData->getData();
        // TODO: ask about using DataItem* b/c it is an abstract class, so should either be DataLevel or DataInfo?
        DataItem* originalGraphInfo = originalRootGraphLevel->next();
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, true, true);

        transformedGraphInfo->addOpt(COL_TILE_DOPT, 65000.0); // TODO: change 65000 to match user input
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
        DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
        DataNode* transformedGraph = new DataNode("Graph-Tile", graphData->getIType(), graphData->getNType(),
            graphData->getVType(), transformedRootGraphLevel);

        dataNodeMap["Transform-Tile"] = transformedGraph;

        // Association between transformed graph and features
        RelationEdge* trGraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, featData, ROWS_RELATION);
        TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
        tileTransformation->addParam("65000");
        TransformEdge* graphTrgraph = new TransformEdge(graphData, transformedGraph);
        graphTrgraph->addTransformation(tileTransformation);
        transforms.push_back(graphTrgraph);

        free($1);
        free($5);
    }
;
// whatever is last in the grammar holds the TrainingLoopNode()
// keep on copying to the newest one and freeing the old one
algorithm : { $$ = NULL; }
    | statement algorithm // so far trainingLoopNode used regardless if there is layers or not
    {
        if ($2 == NULL){
            $$ = new TrainingLoopNode(100); // default for this one
        }
        else{
            $$ = $2;
        }
        if ($1 != NULL){
            $$->addLoopNode($1);
        }
    }
    | layers model algorithm // at least one layer needs to be defined
    {
        if ($1 != NULL){
            $$ = $1;
        }
        else{
            $$ = new TrainingLoopNode(100);
        }
        if ($3 != NULL){
            for (ForwardNode* forwardNode : *($3->getLoopNodes())){
                $$->addLoopNode(forwardNode);
            }
        }
    }
;
statements : { $$ = NULL; }
    | statements statement 
    {
        if ($1){
            $$ = $1;
        }
        else{
            $$ = new TrainingLoopNode(100);
        }
        if ($2){
            $$->addLoopNode($2);
        }
    }
;
// TODO: finish rest of statement rules
statement : IDENTIFIER ASSIGN gnn_op SEMICOLON
    {
        // TODO: add some code to verify the aggregate function name matches
        if (string($3) == "aggregate"){ // aggregate operation
            $$ = new ForwardNode(AGGREGATE_NODE, MUL_SUM_OP);
            DataInfo* outputInfo = new DataInfo(RM_DTYPE);
            outputInfo->setDims(-1, 605); // -1=N=232965, the number of nodes in the graph
            DataLevel* rootOutputLevel = new DataLevel(outputInfo, true);
            DataNode* outputData = new DataNode("Out1", INT32, INT32, F32, rootOutputLevel);
            dataNodeMap["Output-Aggregate"] = outputData;
            
            $$->addInputData(dataNodeMap["Feat"]);
            $$->addInputData(dataNodeMap["Transform-Tile"]);
            $$->addInputData(outputData);

            // Relation (dependency) between features and aggregated output
            RelationEdge* inOutAggrRelationFeat = new RelationEdge(dataNodeMap["Feat"], ALL_RELATION, outputData, ALL_RELATION);
            RelationEdge* inOutAggrRelationGraph = new RelationEdge(dataNodeMap["Transform-Tile"], ALL_RELATION, outputData, ALL_RELATION);
            dependencies.push_back(inOutAggrRelationFeat);
            dependencies.push_back(inOutAggrRelationGraph);
            
        }
        else if (string($3) == "ffn"){ // weight operation
            $$ = new ForwardNode(UPDATE_NODE, FFN_OP);
            // weight as matrix in DIR
            DataInfo* weightInfo = new DataInfo(RM_DTYPE);
            weightInfo->setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
            DataLevel* weightLevel = new DataLevel(weightInfo, true);
            DataNode* weightData = new DataNode("Weight1", INT32, INT32, F32, weightLevel);
            dataNodeMap["Weight1"] = weightData;

            // Res DIR
            DataInfo* resInfo = new DataInfo(RM_DTYPE);
            resInfo->setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
            DataLevel* rootResLevel = new DataLevel(resInfo, true);
            DataNode* resData = new DataNode("Res1", INT32, INT32, F32, rootResLevel);
            dataNodeMap["Res1"] = resData;
            $$->addInputData(dataNodeMap["Output-Aggregate"]);
            $$->addInputData(weightData);
            $$->addOutputData(resData);

            // Relation (dependency) between weight and features 
            RelationEdge* inOutWeightDepRelationFeat = new RelationEdge(dataNodeMap["Output-Aggregate"], ALL_RELATION, resData, ALL_RELATION);
            RelationEdge* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
            dependencies.push_back(inOutWeightDepRelationFeat);
            dependencies.push_back(inOutWeightDepRelationWeight);
            // Relation (association) between aggregate node and weight
            RelationEdge* inOutWeightAssociation = new RelationEdge(dataNodeMap["Output-Aggregate"], ROWS_RELATION, weightData, COLS_RELATION);
            associations.push_back(inOutWeightAssociation);
        }
        else if (string($3) == "relu"){
            $$ = NULL;
        }
        free($1);
        free($3);
    }
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON
    {
        $$ = NULL;
        free($1);
        free($3);
    }
    | IDENTIFIER ASSIGN function_init SEMICOLON
    {
        $$ = NULL;
        free($1);
    }
;
layers : layer_def { $$ = $1; } // layers are just one or more layer_def
    | layers layer_def // not gonna worry about multiple layer defs for now
    {}
;
layer_def : IDENTIFIER ASSIGN LAYER LPAREN args RPAREN LBRACE statements RBRACE
    {
        $$ = new TrainingLoopNode(100);
        if ($8 != NULL){
            $$ = $8;
        }
        else{
            $$ = new TrainingLoopNode(100);
        }
        free($1);
    }
;
// TODO: add models rule for multiple model_def (still deciding if necessary)
model : model_def model_init model_uses {}
;
model_def : IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE {}
;
layer_inits : { $$ = NULL; }
    | layer_inits layer_init {}
;
layer_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { free($1); free($3); }
;
model_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { free($1); free($3); }
;
model_uses : { $$ = NULL; }
    | model_uses model_use {}
;
model_use : IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON { free($1); free($3); }
    | IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON { free($1); }
;
gnn_op : 
    data_var op data_var { free($1); free($3); }
    | function 
    {
        $$ = $1;
    }
;
function : IDENTIFIER LPAREN data_var COMMA data_var RPAREN
    {
        $$ = strdup("aggregate");
        free($1);
        free($3);
        free($5);
    }
    | DSL_DOT update_op 
    {
        $$ = $2;
    }
;
update_op : FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN
    {
        $$ = strdup("ffn");
        free($3);
        free($6);
    }
    | RELU LPAREN data_var RPAREN
    {
        $$ = strdup("relu");
        free($3);
    }
;
// TODO: finish data var, it can be a variable, but also something like G.node.feats
// so IDENTIFIER DOT NODE DOT FEATS and all different combinatioins
data_var : IDENTIFIER
    {
    }
    | IDENTIFIER DOT FEAT_ATTR
    {
        $$ = strdup("feats");
        free($1);
    }
    | IDENTIFIER DOT GRAPH_ATTR
    {
        $$ = strdup("graphs");
        free($1);
    }
    | IDENTIFIER DOT LABEL_ATTR
    {
        $$ = strdup("label");
        free($1);
    }
    | data_var DOT SIZE_FN LPAREN RPAREN
    {
        $$ = strdup("size");
        free($1);
    }
;
function_init : AGGR_INIT LPAREN FN_ARG ASSIGN DSL_FN semiring_op RPAREN
    {}
;
semiring_op : MUL_SUM
    {}
;
op : PLUS {} | MINUS {} | MULTIPLY {} | DIVIDE {}
;
train_args : { $$ = NULL; }
    | train_args train_arg
    {}
;
train_arg : ITERS ASSIGN INTEGER
    {
        free($3);
    }
    | ITERS ASSIGN INTEGER COMMA
    { free($3); }
    | VAL_STEP ASSIGN INTEGER
    { free($3); }
    | VAL_STEP ASSIGN INTEGER COMMA
    { free($3); }
;
args : { $$ = NULL; }
    | args arg
    {

    }
;
arg : INTEGER COMMA | INTEGER
    {
        free($1);
    }
    | NULL_KEY COMMA | NULL_KEY
    {}
    | data_var COMMA | data_var
    {
        free($1);
    }
;
string : QUOTE IDENTIFIER QUOTE
    {
        $$ = (char*) malloc(strlen($2) + 2);
        sprintf($$, "%s", $2);
        free($2);
    }
;
// TODO: finish data_transform
%%

int main(int argc, char** argv){
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
    cout << "PROGRAM\n";
    for (auto a : program){
        cout << a << '\n';
        delete a;
    }
    cout << "DEPENDENCIES\n";
    for (auto a : dependencies){
        cout << a << '\n';
        delete a;
    }
    cout << "ASSOCIATIONS\n";
    for (auto a : associations){
        cout << a << '\n';
        delete a;
    }
    cout << "TRANSFORMS\n";
    for (auto a : transforms){
        cout << a << '\n';
        delete a;
    }

    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}