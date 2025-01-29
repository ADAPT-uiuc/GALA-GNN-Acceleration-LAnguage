%{
#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include "ir/data.h"
#include "ir/compute.h"
#include "ir/frontendIR.h"
using namespace std;

extern int yydebug;
extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);
vector<CIRNode*> program;
TrainingLoopNode* mainTrainingLoop;
int iters = -1;
int numLayers = 0;
vector<RelationEdge*> dependencies;
vector<RelationEdge*> associations;
vector<TransformEdge*> transforms;
map<string, DataNode*> dataNodeMap;
map<string, vector<ForwardNode*>> computeNodeMap;
map<string, int> trainArgs;

// add an aggregate compute node to the CIR, with the appropriate data nodes in the DIR
// the external input (defaultInput) is the last thing in the trainingNodeLoop
// also the graph (transformed graph) is used as an input, but just for the aggregate node
void addAggregate_CIR(DataNode* defaultInput, DataNode* graph){
    ForwardNode* aggregate = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    DataInfo* outputInfo = new DataInfo(RM_DTYPE);
    outputInfo->setDims(-1, -2); // -1=N=232965, the number of nodes in the graph
    DataLevel* rootOutputLevel = new DataLevel(outputInfo, true);
    DataNode* outputData = new DataNode("res-aggregate", INT32, INT32, F32, rootOutputLevel);
    dataNodeMap["Output-Aggregate"] = outputData;
    
    aggregate->addInputData(defaultInput);
    // below is supposed to be transformed graph, but not sure if it exists yet b/c
    // schedule is at bottom of file (so will be read by parser latere), so will 
    // just add both until a removeInputData() method is created?
    aggregate->addInputData(graph); 
    aggregate->addOutputData(outputData);
    computeNodeMap["aggregate"].push_back(aggregate); 
    mainTrainingLoop->addLoopNode(aggregate);


    // Relation (dependency) between features and aggregated output
    RelationEdge* inOutAggrRelationFeat = new RelationEdge(defaultInput, ALL_RELATION, outputData, ALL_RELATION);
    RelationEdge* inOutAggrRelationGraph = new RelationEdge(graph, ALL_RELATION, outputData, ALL_RELATION);
    dependencies.push_back(inOutAggrRelationFeat);
    dependencies.push_back(inOutAggrRelationGraph);
}
void addFFN_CIR(DataNode* defaultInput){
    ForwardNode* ffn = new ForwardNode(UPDATE_NODE, FFN_OP);
    // weight as matrix in DIR
    DataInfo* weightInfo = new DataInfo(RM_DTYPE);
    weightInfo->setDims(-2, -3); // -1=N=232965, the number of nodes in the graph
    DataLevel* weightLevel = new DataLevel(weightInfo, true);
    DataNode* weightData = new DataNode("Weight1", INT32, INT32, F32, weightLevel);
    dataNodeMap["Weight1"] = weightData;
    // Res DIR
    DataInfo* resInfo = new DataInfo(RM_DTYPE);
    resInfo->setDims(-1, -3); // -1=N=232965, the number of nodes in the graph
    DataLevel* rootResLevel = new DataLevel(resInfo, true);
    DataNode* resData = new DataNode("res-weight", INT32, INT32, F32, rootResLevel);
    dataNodeMap["Res1"] = resData;
    ffn->addInputData(defaultInput);
    ffn->addInputData(weightData);
    ffn->addOutputData(resData);
    computeNodeMap["ffn"].push_back(ffn);
    mainTrainingLoop->addLoopNode(ffn);
    // Relation (dependency) between weight and features 
    RelationEdge* inOutWeightDepRelationFeat = new RelationEdge(defaultInput, ALL_RELATION, resData, ALL_RELATION);
    RelationEdge* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
    dependencies.push_back(inOutWeightDepRelationFeat);
    dependencies.push_back(inOutWeightDepRelationWeight);
    // Relation (association) between aggregate node and weight
    RelationEdge* inOutWeightAssociation = new RelationEdge(defaultInput, ROWS_RELATION, weightData, COLS_RELATION);
    associations.push_back(inOutWeightAssociation);
}
void addNormalization_CIR(DataNode* defaultInput){
    ForwardNode* powerOp = new ForwardNode(POINTWISE, POWER_OP);
	powerOp->addParam("-0.5");
	DataInfo* normInfo = new DataInfo(RM_DTYPE);
	normInfo->setDims(-1, 1);
	DataLevel* rootNormLevel = new DataLevel(normInfo, true);
	DataNode* normData = new DataNode("norm", INT32, INT32, F32, rootNormLevel);
	powerOp->addInputData(defaultInput);
	powerOp->addOutputData(normData);
	mainTrainingLoop->addLoopNode(powerOp);
	RelationEdge* powerOpDegreesDependency = new RelationEdge(defaultInput, ALL_RELATION, normData, ALL_RELATION);
	dependencies.push_back(powerOpDegreesDependency);
}
void addNormCalc_CIR(DataNode* defaultInput, DataNode* featData){
	// 1st normalization calculation
	ForwardNode* normFeat1 = new ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
	DataInfo* normFeat1Info = new DataInfo(RM_DTYPE);
	normFeat1Info->setDims(-1, 602);
	DataLevel* rootNormFeat1Level = new DataLevel(normFeat1Info, true);
	DataNode* normFeat1Data = new DataNode("res-norm", INT32, INT32, F32, rootNormFeat1Level);
	normFeat1->addInputData(defaultInput);
	normFeat1->addInputData(featData);
	normFeat1->addOutputData(normFeat1Data);
	mainTrainingLoop->addLoopNode(normFeat1);
	RelationEdge* normFeat1NormDependency = new RelationEdge(defaultInput, ALL_RELATION, normFeat1Data, ROWS_RELATION);
	dependencies.push_back(normFeat1NormDependency);
	RelationEdge* normFeat1FeatDependency = new RelationEdge(featData, ALL_RELATION, normFeat1Data, ALL_RELATION);
	dependencies.push_back(normFeat1FeatDependency);
	RelationEdge* normFeat1NormFeatAssociation = new RelationEdge(defaultInput, ALL_RELATION, featData, ROWS_RELATION);
	associations.push_back(normFeat1NormFeatAssociation);
}
void addReLU_CIR(DataNode* defaultInput){
    // ReLU operation
	ForwardNode* reluOp = new ForwardNode(POINTWISE, NON_LNR_OP_RELU);
	DataInfo* reluInfo = new DataInfo(RM_DTYPE);
	reluInfo->setDims(-1, 32);
	DataLevel* rootReluLevel = new DataLevel(reluInfo, true);
	DataNode* reluData = new DataNode("res-relu", INT32, INT32, F32, rootReluLevel);
	reluOp->addInputData(defaultInput);
	reluOp->addOutputData(reluData);
	mainTrainingLoop->addLoopNode(reluOp);
	RelationEdge* reluOpOnesDependency = new RelationEdge(defaultInput, ALL_RELATION, reluData, ALL_RELATION);
	dependencies.push_back(reluOpOnesDependency);
}
void addDegrees_CIR(DataNode* defaultInput, DataNode* graph){
    /// Degree op
	// Ones
	// NOTE: In the front-end code there is no need for a user to get the degrees() by getting ones, and doing an
	// aggregation. This is just how it's done under the hood.
	ForwardNode* onesTensorOp = new ForwardNode(POINTWISE, ONES_OP);
	DataInfo* onesInfo = new DataInfo(RM_DTYPE);
	onesInfo->setDims(-1, 1);
	DataLevel* rootOnesLevel = new DataLevel(onesInfo, true);
	DataNode* onesData = new DataNode("ones", INT32, INT32, F32, rootOnesLevel);
	onesTensorOp->addOutputData(onesData);
	mainTrainingLoop->addLoopNode(onesTensorOp);
	//* Dependencies
	RelationEdge* onesTensorGraphAssociation = new RelationEdge(graph, ALL_RELATION, onesData, ROWS_RELATION);
	associations.push_back(onesTensorGraphAssociation);

	// The actual degrees calculation
	ForwardNode* degreesOp = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_DIRECT);
	DataInfo* degreesInfo = new DataInfo(RM_DTYPE);
	degreesInfo->setDims(-1, 1);
	DataLevel* rootDegreesLevel = new DataLevel(degreesInfo, true);
	DataNode* degreesData = new DataNode("degrees", INT32, INT32, F32, rootDegreesLevel);
	degreesOp->addInputData(onesData);
	degreesOp->addInputData(graph);
	degreesOp->addOutputData(degreesData);
	degreesOp->addOpt(COARSE_COPT, 2);
	mainTrainingLoop->addLoopNode(degreesOp);
	RelationEdge* degreesOpOnesDependency = new RelationEdge(onesData, ALL_RELATION, degreesData, ALL_RELATION);
	dependencies.push_back(degreesOpOnesDependency);
	RelationEdge* degreesOpGraphDependency = new RelationEdge(graph, ALL_RELATION, degreesData, ROWS_RELATION);
	dependencies.push_back(degreesOpGraphDependency);
}
%}
%union {
    int ival;
    float fval;
    char *sval;
    FrontendIRNode *irNode;
    ForwardNode* forwardNode;
    TrainingLoopNode* trainingLoopNode;
}
%debug 

%token<sval> IDENTIFIER ASSIGN LOAD;
%token<sval> LPAREN RPAREN SEMICOLON QUOTE COMMENT SET_UNWEIGHTED SET_UNDIRECTED
%token<sval> MODEL_W EVAL TRAIN LAYER LOSS OPTIMIZER ITERS VAL_STEP RMSE_LOSS ADAM_T
%token<sval> AGGR_INIT FN_ARG MUL_SUM DSL_FN DSL_DOT FFN_OUT SIZE_FN 
%token<sval> RELAXNLN QUANT GRAPH_ATTR FEAT_ATTR RELU LABEL_ATTR DEGREE_ATTR NODE_ATTR
%token<sval> RABBIT_REORDER_OP SAMPLE_RANDOM_OP POW
%token<sval> COLTILE AGGR FEAT_SIZE_ASSIGN LABEL_SIZE_ASSIGN COARSEN
%token<sval> INTEGER FLOAT
%token<sval> LBRACE RBRACE LSQBRA RSQBRA DOT COMMA;
%token<sval> IF ELSE DO WHILE;
%token<sval> TR FA;
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

%type <ival> bool
%type <sval> arg train_arg string data_var function update_op gnn_op 
%type <forwardNode> load_dataset function_init statement
%type <trainingLoopNode> algorithm layers layer_def statements
%type <irNode> program schedule args train_args
%type <irNode> layer_inits layer_init data_transform function_transform
%type <irNode> model model_def model_init model_uses model_use 

%%
program : load_dataset algorithm schedules
    {
        program.push_back($1);
        // program.push_back($2);
        program.push_back(mainTrainingLoop);
        /*
        if transformation exists, passed through schedules
        then to the first compute node in training loop add 
        transformed graph as input data
        */
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            // should it be a map of vectors, for all the different aggregate nodes across the layers?
            for (ComputeNode* a : computeNodeMap["aggregate"]){
                a->addInputData(dataNodeMap["TrGraph"]);
            }
            for (ComputeNode* a : computeNodeMap["degreesOp"]){
                a->addInputData(dataNodeMap["TrGraph"]);
            }
            RelationEdge* inOutAggrRelationGraph = new RelationEdge(dataNodeMap["TrGraph"], ALL_RELATION, dataNodeMap["Output-Aggregate"], ALL_RELATION);
            dependencies.push_back(inOutAggrRelationGraph);
        }
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
        featInfo->setDims(-1, -2); 
        DataLevel* rootFeatLevel = new DataLevel(featInfo, true);
        DataNode* featData = new DataNode("Feat", INT32, INT32, F32, rootFeatLevel);

        dataNodeMap["Graph"] = graphData;
        dataNodeMap["Feat1"] = featData; // for future use (e.g. TrainingLoop is in another rule)

        // Relation (association) between graph and features
        RelationEdge* graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(graphFeatAssociation);

        $$->addOutputData(featData);
        $$->addOutputData(graphData);
        
        free($1);
        free($5);
    }
;
// whatever is last in the grammar holds the TrainingLoopNode()
// keep on copying to the newest one 
algorithm : { $$ = NULL; }
    | statement algorithm // so far trainingLoopNode used regardless if there is layers or not
    {
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
    | layers model // algorithm supposed to be here but creates lot of ambiguity
    {              // so for now just one layer+model then schedules
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
;
statements : { $$ = NULL; }
    | statements statement 
    {
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
;
// TODO: finish rest of statement rules
statement : IDENTIFIER ASSIGN gnn_op SEMICOLON
    {
        // TODO: add some code to verify the aggregate function name matches
        if (string($3) == "aggregate"){ // aggregate operation
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
                // so far everything has *exactly* one output data (not more not less), so indexing 0 
                // should be okay for now
            }
            addAggregate_CIR(defaultInput, dataNodeMap["Graph"]); // aggregate node always include graph seperate
        }
        else if (string($3) == "ffn"){ // weight operation
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
                // so far everything has *exactly* one output data (not more not less), so indexing 0 
                // should be okay for now
            }
            addFFN_CIR(defaultInput);
        }
        else if (string($3) == "pow"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
                // so far everything has *exactly* one output data (not more not less), so indexing 0 
                // should be okay for now
            }
            addNormalization_CIR(defaultInput);
        }
        else if (string($3) == "normalization"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
                // so far everything has *exactly* one output data (not more not less), so indexing 0 
                // should be okay for now
            }
            addNormCalc_CIR(defaultInput, dataNodeMap["Feat1"]);
        }
        else if (string($3) == "relu"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
                // so far everything has *exactly* one output data (not more not less), so indexing 0 
                // should be okay for now
            }
            addReLU_CIR(defaultInput);
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
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR DOT DEGREE_ATTR LPAREN RPAREN SEMICOLON
    {
        // TODO: Replace dataNodeMap["Graph"] with TransformedGraph if exists for whatever is necessary in this section
        DataNode* defaultInput;
        if (mainTrainingLoop->getLoopNodeNum() == 0){
            defaultInput = NULL;
        }
        else{
            int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
            defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            // so far everything has *exactly* one output data (not more not less), so indexing 0 
            // should be okay for now
        }
        if (dataNodeMap.find("TrGraph") == dataNodeMap.end()){
            addDegrees_CIR(defaultInput, dataNodeMap["Graph"]);
        }
        else{
            addDegrees_CIR(defaultInput, dataNodeMap["TrGraph"]);
        }
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
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
        // if ($8 != NULL){
        //     $$ = $8;
        // }
        // else{
        //     $$ = new TrainingLoopNode(iters);
        // }
        // free($1);
    }
;
// TODO: add models rule for multiple model_def (still deciding if necessary)
model : model_def model_init model_uses {}
;
model_def : IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE {
    int n = mainTrainingLoop->getLoopNodeNum();
    for (int i = 1; i < numLayers; i++){
        int n_i = mainTrainingLoop->getLoopNodeNum();
        DataNode* lastData = mainTrainingLoop->getNode(n_i-1)->getOutput(0);
        for (int j = 0; j < n; j++){
            ForwardNode* curNode = mainTrainingLoop->getNode(j);
            string type = curNode->getOutput(0)->getName();
            if (type == "ones" || type == "degrees"){
                lastData = curNode->getOutput(0);
                continue;
            }
            else if (type == "norm"){
                addNormalization_CIR(lastData);
            }
            else if (type == "res-norm"){
                addNormCalc_CIR(lastData, dataNodeMap["Feat1"]);
            }
            else if (type == "res-aggregate"){
                addAggregate_CIR(lastData, dataNodeMap["Graph"]);
            }
            else if (type == "res-weight"){
                addFFN_CIR(lastData);
            }
            else if (type == "res-relu"){
                addReLU_CIR(lastData);
            }
            lastData = curNode->getOutput(0);
        }
    }
}
;
layer_inits : { $$ = NULL; }
    | layer_inits layer_init {}
;
layer_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { numLayers++; free($1); free($3); }
;
model_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { free($1); free($3); }
;
model_uses : model_use model_use { $$ = NULL; }
    /* | model_uses model_use {}  this rule creating a lot of ambiguity syntax errors, 
        avoiding more than two model_uses for now*/ 
;
model_use : IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON { free($1); free($3); }
    | IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON { free($1); }
;
gnn_op : data_var op data_var { $$ = strdup("normalization"); free($1); free($3); }
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
    | POW LPAREN data_var COMMA FLOAT RPAREN
    {
        // TODO: use the float in the calculation, now just hardcoded to -0.5
        $$ = strdup("pow");
        free($3);
    }
;
schedules : schedule {}
    | schedules schedule
    {

    }
;
schedule : data_transform
    {
    }
    | function_transform
    {
    }
;
data_transform : data_var ASSIGN data_var DOT SET_UNDIRECTED LPAREN bool RPAREN SEMICOLON
    {
        // if transformed graph already exists, then modify that as well
        // always modify the original graph
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setDirected(!$7);
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setDirected(!$7);
    }
    | data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON
    {
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setWeighted(!$7);
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setWeighted(!$7);
    }
    | FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat1"]->getData()->next());
        DataInfo* outAggrInfo = dynamic_cast<DataInfo*>(dataNodeMap["Output-Aggregate"]->getData()->next());
        int dimRow = featInfo->getDimRow();
        featInfo->setDims(dimRow, atoi($3));
        outAggrInfo->setDims(dimRow, atoi($3));
        free($3);
    }
    | LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON 
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat1"]->getData()->next());
        DataInfo* weightInfo = dynamic_cast<DataInfo*>(dataNodeMap["Weight1"]->getData()->next());
        DataInfo* resInfo = dynamic_cast<DataInfo*>(dataNodeMap["Res1"]->getData()->next());
        int featDimRow = featInfo->getDimRow();
        int featDimCol = featInfo->getDimCol();
        weightInfo->setDims(featDimCol, atoi($3));
        resInfo->setDims(featDimRow, atoi($3));
        free($3);
    }
    | data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON 
    {
        // actually creating new DIR
        DataLevel* originalRootGraphLevel = dataNodeMap["Graph"]->getData();
        // TODO: ask about using DataItem* b/c it is an abstract class, so should either be DataLevel or DataInfo?
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(originalRootGraphLevel->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), originalGraphInfo->getWeighted());
        transformedGraphInfo->addOpt(COL_TILE_DOPT, atoi($7));
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
        DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
        DataNode* transformedGraph = new DataNode("Graph-Tile", dataNodeMap["Graph"]->getIType(), dataNodeMap["Graph"]->getNType(),
        dataNodeMap["Graph"]->getVType(), transformedRootGraphLevel);

        dataNodeMap["TrGraph"] = transformedGraph;

        // Association between transformed graph and features
        RelationEdge* trGraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, dataNodeMap["Feat"], ROWS_RELATION);
	    associations.push_back(trGraphFeatAssociation);
        // Transformation between original graph and new one
        TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
        tileTransformation->addParam($7);
        TransformEdge* graphTrgraph = new TransformEdge(dataNodeMap["Graph"], transformedGraph);
        graphTrgraph->addTransformation(tileTransformation);
        transforms.push_back(graphTrgraph);
            
        RelationEdge* inOutAggrRelationTrGraph = new RelationEdge(transformedGraph, ALL_RELATION, dataNodeMap["Output-Aggregate"], ALL_RELATION);
        dependencies.push_back(inOutAggrRelationTrGraph);
        free($3);
    }
;
function_transform : data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON
    {
        if (computeNodeMap.find("aggregate") != computeNodeMap.end()){
            for (ForwardNode* a : computeNodeMap["aggregate"]){
                a->addOpt(COARSE_COPT, atoi($7));
            }
        }
        else{
            cout << "error\n";
        }
        free($7);
    }
;
// TODO: finish data var, it can be a variable, but also something like G.node.feats
// so IDENTIFIER DOT NODE DOT FEATS and all different combinatioins
data_var : IDENTIFIER
    {
    }
    | data_var DOT NODE_ATTR
    {
        if ($1 == "feats"){
            $$ = $1;
        }
        $$ = strdup("node");
        free($1);
    }
    | data_var DOT FEAT_ATTR
    {
        $$ = strdup("feats");
        free($1);
    }
    | data_var DOT GRAPH_ATTR
    {
        $$ = strdup("graphs");
        free($1);
    }
    | data_var DOT LABEL_ATTR
    {
        $$ = strdup("label");
        free($1);
    }
    | data_var DOT SIZE_FN LPAREN RPAREN
    {
        $$ = strdup("size");
        free($1);
    }
    | data_var DOT DEGREE_ATTR LPAREN RPAREN
    {
        $$ = strdup("degrees");
        free($1);
    }
;
function_init : AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN
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
        trainArgs["iters"] = atoi($3);
        free($3);
    }
    | ITERS ASSIGN INTEGER COMMA
    { trainArgs["iters"] = atoi($3); free($3); }
    | VAL_STEP ASSIGN INTEGER
    { trainArgs["val_step"] = atoi($3); free($3); }
    | VAL_STEP ASSIGN INTEGER COMMA
    { trainArgs["val_step"] = atoi($3); free($3); }
;
args : { $$ = NULL; }
    | args arg
    {
    }
;
arg : INTEGER COMMA { free($1); } | INTEGER
    {
        free($1);
    }
    | NULL_KEY COMMA | NULL_KEY
    {}
    | data_var COMMA { free($1); } | data_var
    {
        free($1);
    }
    | DSL_DOT RELU | DSL_DOT RELU COMMA
    {

    }
;
bool : TR { $$ = 1; } | FA { $$ = 2; };
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
    yydebug = 0;
    mainTrainingLoop = new TrainingLoopNode(iters == -1 ? 100 : iters);
    yyparse();
    cout << "PROGRAM (CIR Nodes): " << program.size() << '\n';
    cout << "DEPENDENCIES " << dependencies.size() << '\n';
    cout << "ASSOCIATIONS " << associations.size() << '\n';
    cout << "TRANSFORMS " << transforms.size() << '\n';

    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}