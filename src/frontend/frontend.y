%{
#include <iostream>
#include <string.h>
#include <vector>
#include <cstring>
#include <map>
#include "../ir/data.h"
#include "../ir/compute.h"
#include "../ir/frontend_metadata.h"
using namespace std;

extern int yydebug;
extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);

extern ModelConfig m1; // just considering one model per input file for now
int debug = 0;

DataNode* normData; // very temp solution, find fix asap

extern vector<CIRNode*> program;
extern vector<RelationEdge*> dependencies;
extern vector<RelationEdge*> associations;
extern vector<TransformEdge*> transforms;
%}

%union {
    int ival;
    float fval;
    char* sval;
    LayerOpType ltype;
    void* vval;
}
%debug

%token<sval> IDENTIFIER ASSIGN LOAD;
%token<sval> LPAREN RPAREN SEMICOLON QUOTE SET_UNWEIGHTED SET_UNDIRECTED
%token<sval> MODEL_W EVAL TRAIN LAYER ITERS VAL_STEP 
%token<sval> AGGR_INIT FN_ARG MUL_SUM DSL_DOT FFN_OUT SIZE_FN 
%token<sval> GRAPH_ATTR FEAT_ATTR RELU LABEL_ATTR DEGREE_ATTR NODE_ATTR LEAKY_RELU
%token<sval> POW SCALAR_INIT
%token<sval> COLTILE FEAT_SIZE_ASSIGN LABEL_SIZE_ASSIGN COARSEN SRC_ATTR DST_ATTR;
%token<sval> INTEGER FLOAT SOFTMAX INIT_WEIGHT;
%token<sval> LBRACE RBRACE DOT COMMA;
%token<sval> TR FA;
%token<sval> PLUS MINUS MULTIPLY DIVIDE;
%token<sval> FFN NULL_KEY
%token<sval> LOSS OPTIMIZER RMSE_LOSS ADAM_T DSL_FN RELAXNLN QUANT RABBIT_REORDER_OP SAMPLE_RANDOM_OP AGGR LSQBRA RSQBRA IF ELSE DO WHILE NOT AND OR NOTEQ EQ GREATER LESS GREATEREQ LESSEQ DATASET NONLN SENSEI_OP INT NEW

%type <ival> bool op
%type <ltype> function update_op gnn_op
%type <sval> arg train_arg string data_var 
%type <vval> load_dataset function_init statement
%type <vval> algorithm layers layer_def statements
%type <vval> program schedule args train_args
%type <vval> layer_inits layer_init data_transform function_transform
%type <vval> model model_def model_init model_uses model_use 

%%
program : load_dataset algorithm schedules {}
;
load_dataset : IDENTIFIER ASSIGN LOAD LPAREN string RPAREN SEMICOLON 
    { m1.dataset_name = $5; }
;
algorithm : { }
    | statement algorithm {}
    | layers model {}
;
statements : { }
    | statements statement 
    {}
;
statement : IDENTIFIER ASSIGN gnn_op SEMICOLON
    {
        if ($3 == GET_DEGREES){
            if (debug == 2) cout << "layer op - get degrees\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == GET_NORMALIZATION){
            if (debug == 2) cout << "layer op - get normalization\n";
            m1.layer_operations.push_back($3);

        }
        else if ($3 == MULT_NORM_RES){
            if (debug == 2) cout << "layer op - mult norm res\n";
            m1.layer_operations.push_back($3);

        }
        else if ($3 == MESSAGE_PASSING_AGGREGATE){
            if (debug == 2) cout << "layer op - aggregate\n";
            m1.layer_operations.push_back($3);

        }
        else if ($3 == FEED_FORWARD_NN){
            if (debug == 2) cout << "layer op - ffn\n";
            m1.layer_operations.push_back($3);

        }
        else if ($3 == NON_LINEARITY){
            if (debug == 2) cout << "layer op - nonln\n";
            m1.layer_operations.push_back($3);

        }
    }
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON
    {
    }
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR DOT DEGREE_ATTR LPAREN RPAREN SEMICOLON
    {
        if (debug == 2) cout << "layer op - get degrees\n";
        m1.layer_operations.push_back(GET_DEGREES);

    }
    | IDENTIFIER ASSIGN function_init SEMICOLON
    {
    }
;
layers : layer_def {} 
    | layers layer_def 
    {}
;
layer_def : IDENTIFIER ASSIGN LAYER LPAREN args RPAREN LBRACE statements RBRACE
    {}
;
// TODO: add models rule for multiple model_def (still deciding if necessary)
model : model_def model_init model_uses {}
;
model_def : IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE {}
;
layer_inits : { }
    | layer_inits layer_init {}
;
layer_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { m1.num_layers++; }
;
model_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { }
;
model_uses : model_use model_use {  }
    /* | model_uses model_use {}  this rule creating a lot of ambiguity syntax errors, 
        avoiding more than two model_uses for now*/ 
;
model_use : IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON {  }
    | IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON {  }
;
gnn_op : data_var op data_var
    {
        if ($2 == 3){
            $$ = MULT_NORM_RES;
        }
    }
    | function 
    { $$ = $1; }
    | function op data_var
    {}
;
function : IDENTIFIER LPAREN data_var COMMA data_var RPAREN
    {
        // for now we basically know this is an aggregate (but will need to update later)
        $$ = MESSAGE_PASSING_AGGREGATE;

    }
    | IDENTIFIER LPAREN data_var RPAREN
    {
        // similarly know that this is relu
        $$ = NON_LINEARITY;
    }
    | DSL_DOT update_op 
    {
        $$ = $2;
    }
;
update_op : FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN
    { $$ = FEED_FORWARD_NN; }
    | RELU LPAREN data_var RPAREN
    { $$ = NON_LINEARITY; }
    | LEAKY_RELU LPAREN data_var op data_var RPAREN
    { } // do this later
    | POW LPAREN data_var COMMA FLOAT RPAREN
    { $$ = GET_NORMALIZATION; }
    | SCALAR_INIT LPAREN INTEGER RPAREN
    {}
    | SOFTMAX LPAREN data_var RPAREN
    {}
    | INIT_WEIGHT LPAREN RPAREN
    {}
;
schedules : schedule {}
    | schedules schedule
    {}
;
schedule : data_transform
    {
    }
    | function_transform
    {
    }
;
// this is where todo all graph transform, data transform 
data_transform : data_var ASSIGN data_var DOT SET_UNDIRECTED LPAREN bool RPAREN SEMICOLON
    {  m1.addGraphTransformation(UNDIRECTED, (float) !$7);  }
    | data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON
    {  m1.addGraphTransformation(UNWEIGHTED, (float) !$7); }
    | FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON
    {  m1.addGraphTransformation(FEAT_SIZE, atof($3));  }
    | LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON 
    {  m1.addGraphTransformation(LABEL_SIZE, atof($3));  }
    | data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON 
    {  m1.addDataTransformation(COL_TILE, atof($7));  }
;
// this is where to do compute transform
function_transform : data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON
    {
        m1.addComputeTransformation(COARSE, atof($7));
    }
;
// TODO: finish data var, it can be a variable, but also something like G.node.feats
// so IDENTIFIER DOT NODE DOT FEATS and all different combinatioins
data_var : IDENTIFIER
    {
    }
    | data_var DOT NODE_ATTR
    {
        if (strcmp($1, "feats") == 0){
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
    | data_var DOT SRC_ATTR
    {
        $$ = strdup("src_nodes");
        free($1);
    }
    | data_var DOT DST_ATTR
    {
        $$ = strdup("dst_nodes");
        free($1);
    }
;
function_init : AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN
    {}
;
semiring_op : MUL_SUM
    {}
;
op : PLUS { $$ = 1; } | MINUS { $$ = 2; } | MULTIPLY { $$ = 3; } | DIVIDE { $$ = 4;}
;
train_args : {  }
    | train_args train_arg
    {}
;
train_arg : ITERS ASSIGN INTEGER
    {  m1.iterations = atoi($3); free($3); }
    | ITERS ASSIGN INTEGER COMMA
    { m1.iterations = atoi($3); free($3); }
    | VAL_STEP ASSIGN INTEGER // do later
    { m1.validation_step = atoi($3); free($3); }
    | VAL_STEP ASSIGN INTEGER COMMA
    { m1.validation_step = atoi($3); free($3); }
;
args : {  }
    | args arg {}
;
arg : INTEGER COMMA { m1.output_input_classes = atof($1); } | INTEGER
    { m1.output_input_classes = atof($1); }
    | NULL_KEY COMMA | NULL_KEY {}
    | data_var COMMA { } | data_var {}
    | DSL_DOT RELU | DSL_DOT RELU COMMA  {}
;
bool : TR { $$ = 1; } | FA { $$ = 0; };
string : QUOTE IDENTIFIER QUOTE
    {
        $$ = (char*) malloc(strlen($2) + 2);
        sprintf($$, "%s", $2);
        free($2);
    }
;
%%

DataNode* createDataNode(int rm_dtype, bool isDirected, bool isWeighted, pair<int,int> infoDims, bool levelIndependence, string name, NumTypes indexType, NumTypes edgeType, NumTypes valueType){
    DataInfo* info = new DataInfo(RM_DTYPE, isDirected, isWeighted);
    info->setDims(infoDims.first, infoDims.second);
    DataLevel* level = new DataLevel(info, levelIndependence);
    DataNode* data = new DataNode(name, indexType, edgeType, valueType, level);
    return data;
}
DataNode* addDegrees_CIR(DataNode* graphData, TrainingLoopNode* trainingLoop){
    if (debug == 2) cout << "degrees\n";
	ForwardNode* onesTensorOp = new ForwardNode(POINTWISE, ONES_OP);
    DataNode* onesData = createDataNode(RM_DTYPE, false, false, {-1, 1}, true, "ones", INT32, INT32, F32);
	onesTensorOp->addOutputData(onesData);
	trainingLoop->addLoopNode(onesTensorOp);
	//* Dependencies
	RelationEdge* onesTensorGraphAssociation = new RelationEdge(graphData, ALL_RELATION, onesData, ROWS_RELATION);
	associations.push_back(onesTensorGraphAssociation);

	// The actual degrees calculation
	ForwardNode* degreesOp = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_DIRECT);
	DataInfo* degreesInfo = new DataInfo(RM_DTYPE);
	degreesInfo->setDims(-1, 1);
	DataLevel* rootDegreesLevel = new DataLevel(degreesInfo, true);
	DataNode* degreesData = new DataNode("degrees", INT32, INT32, F32, rootDegreesLevel);
	degreesOp->addInputData(onesData);
	degreesOp->addInputData(graphData);
	degreesOp->addOutputData(degreesData);
	degreesOp->addOpt(COARSE_COPT, 2);
	trainingLoop->addLoopNode(degreesOp);
	RelationEdge* degreesOpOnesDependency = new RelationEdge(onesData, ALL_RELATION, degreesData, ALL_RELATION);
	dependencies.push_back(degreesOpOnesDependency);
	RelationEdge* degreesOpGraphDependency = new RelationEdge(graphData, ALL_RELATION, degreesData, ROWS_RELATION);
	dependencies.push_back(degreesOpGraphDependency);
    return degreesData;
}
DataNode* addNormalization_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop){
    if (debug == 2) cout << "normalization setup\n";
    ForwardNode* powerOp = new ForwardNode(POINTWISE, POWER_OP);
	powerOp->addParam("-0.5"); // hardcoded for now
    DataNode* normData = createDataNode(RM_DTYPE, false, false, {-1, 1}, true, "norm", INT32, INT32, F32);
	powerOp->addInputData(prevData);
	powerOp->addOutputData(normData);
	trainingLoop->addLoopNode(powerOp);
	RelationEdge* powerOpDegreesDependency = new RelationEdge(prevData, ALL_RELATION, normData, ALL_RELATION);
	dependencies.push_back(powerOpDegreesDependency);
    return normData;
}
DataNode* addNormCalc_CIR(DataNode* normData, DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum, bool featInput){ // prevData is either feat or res
    if (debug == 2) cout << "normalization-calculation\n";
	// 1st normalization calculation
	ForwardNode* normFeat1 = new ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
    pair<int,int> normFeat1Data_inputDim = {-1, featInput ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes};
    DataNode* normFeat1Data = createDataNode(RM_DTYPE, false, false, normFeat1Data_inputDim, true, "res", INT32, INT32, F32);
	normFeat1->addInputData(normData);
	normFeat1->addInputData(prevData);
	normFeat1->addOutputData(normFeat1Data);
	trainingLoop->addLoopNode(normFeat1);
	RelationEdge* normFeat1NormDependency = new RelationEdge(normData, ALL_RELATION, normFeat1Data, ROWS_RELATION);
	dependencies.push_back(normFeat1NormDependency);
	RelationEdge* normFeat1FeatDependency = new RelationEdge(prevData, ALL_RELATION, normFeat1Data, ALL_RELATION);
	dependencies.push_back(normFeat1FeatDependency);
	RelationEdge* normFeat1NormFeatAssociation = new RelationEdge(normData, ALL_RELATION, prevData, ROWS_RELATION);
	associations.push_back(normFeat1NormFeatAssociation);
    return normFeat1Data;

}
DataNode* addAggregate_CIR(DataNode* prevData, DataNode* graphData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "aggregate" << '\n';
    ForwardNode* aggregate = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    pair<int,int> outputData_inputDim = {-1, (layerNum == 0) ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes};
    DataNode* outputData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, "res", INT32, INT32, F32);
    
    aggregate->addInputData(prevData);
    aggregate->addInputData(graphData); 
    aggregate->addOutputData(outputData);
    trainingLoop->addLoopNode(aggregate);

    // Relation (dependency) between features and aggregated output
    RelationEdge* inOutAggrRelationFeat = new RelationEdge(prevData, ALL_RELATION, outputData, ALL_RELATION);
    RelationEdge* inOutAggrRelationGraph = new RelationEdge(graphData, ALL_RELATION, outputData, ALL_RELATION);
    dependencies.push_back(inOutAggrRelationFeat);
    dependencies.push_back(inOutAggrRelationGraph);
    return outputData;

}
DataNode* addFFN_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "ffn" << '\n';
    ForwardNode* ffn = new ForwardNode(UPDATE_NODE, FFN_OP);
    // weight as matrix in DIR
    string weightNum = "weight" + to_string(layerNum+1);
    pair<int,int> weightInputDim;
    pair<int,int> resInputDim;
    if (layerNum == 0){
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes};
        resInputDim = {-1, m1.output_input_classes};
    }
    else{
        weightInputDim = {m1.output_input_classes, m1.graph_transformations[LABEL_SIZE]};
        resInputDim = {-1, m1.graph_transformations[LABEL_SIZE]};
    }
    DataNode* weightData = createDataNode(RM_DTYPE, false, false, weightInputDim, true, weightNum, INT32, INT32, F32);
    // Res DIR
    DataNode* resData = createDataNode(RM_DTYPE, false, false, resInputDim, true, "res", INT32, INT32, F32);
    ffn->addInputData(prevData);
    ffn->addInputData(weightData);
    ffn->addOutputData(resData);
    trainingLoop->addLoopNode(ffn);
    // Relation (dependency) between weight and features 
    RelationEdge* inOutWeightDepRelationFeat = new RelationEdge(prevData, ALL_RELATION, resData, ALL_RELATION);
    RelationEdge* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
    dependencies.push_back(inOutWeightDepRelationFeat);
    dependencies.push_back(inOutWeightDepRelationWeight);
    // Relation (association) between aggregate node and weight
    RelationEdge* inOutWeightAssociation = new RelationEdge(prevData, ROWS_RELATION, weightData, COLS_RELATION);
    associations.push_back(inOutWeightAssociation);
    return resData;

}
DataNode* addReLU_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "relu\n";
    // ReLU operation
	ForwardNode* reluOp = new ForwardNode(POINTWISE, NON_LNR_OP_RELU);
    pair<int,int> reluData_inputDim = {-1, (layerNum == 0) ? m1.output_input_classes : m1.graph_transformations[LABEL_SIZE]};
    DataNode* reluData = createDataNode(RM_DTYPE, false, false, reluData_inputDim, true, "res", INT32, INT32, F32);
	reluOp->addInputData(prevData);
	reluOp->addOutputData(reluData);
	trainingLoop->addLoopNode(reluOp);
	RelationEdge* reluOpOnesDependency = new RelationEdge(prevData, ALL_RELATION, reluData, ALL_RELATION);
	dependencies.push_back(reluOpOnesDependency);
    return reluData;
}

/*
purpose:
 - looks through model config layer_transformations and makes the appropriate nodes and edges accordingly
parameters:
 - isFirstLayer: stuff like degreesOp and powerOp are only created for first layer, if not first skip those 
 - connectingNodes: empty for first layer, but for future layers its the inputs to that layer
 - graph: either un-transformed or transformed graph to be put in as one of the connecting nodes
 - trainingLoop: where the nodes will be added
return:
 - output connecting nodes, whats fed in to the next layer (if there is one)
 */
DataNode* addLayer(int layerNum, DataNode* connectNode, DataNode* graphData, DataNode* featData, TrainingLoopNode* trainingLoop){
    DataNode* prevData = connectNode;
    for (LayerOpType t : m1.layer_operations){
        switch (t){
            case GET_DEGREES: // if first layer, this is first for gcn
                if (layerNum == 0){
                    prevData = addDegrees_CIR(graphData, trainingLoop);
                }
                break;
            case GET_NORMALIZATION:
                if (layerNum == 0){
                    normData = addNormalization_CIR(prevData, trainingLoop);
                }
                break;
            case MULT_NORM_RES:
                if (trainingLoop->getLoopNodeNum() < 5) // temporary fix to see if use featData or resData
                    prevData = addNormCalc_CIR(normData, featData, trainingLoop, layerNum, true);
                else
                    prevData = addNormCalc_CIR(normData, prevData, trainingLoop, layerNum, false);
                break;
            case MESSAGE_PASSING_AGGREGATE:
                prevData = addAggregate_CIR(prevData, graphData, trainingLoop, layerNum);
                break;
            case FEED_FORWARD_NN:
                prevData = addFFN_CIR(prevData, trainingLoop, layerNum);
                break;
            case NON_LINEARITY:
                prevData = addReLU_CIR(prevData, trainingLoop, layerNum);
                break;
        }
    }
    return prevData;
}
void generate_ir(){
    DataNode* graphData; 
    DataNode* featData;
    RelationEdge* graphFeatAssociation;
    if (m1.dataset_name != "\0"){ // load dataset
        if (debug == 2) cout << "load dataset section with name " << m1.dataset_name << "\n";
        auto loadDataset = ForwardNode(POINTWISE, LOAD_OP);
        loadDataset.addParam(m1.dataset_name);
        graphData = createDataNode(CSR_STYPE, true, true, {0,0}, true, "adj0", INT32, INT32, F32);
        featData = createDataNode(RM_DTYPE, false, false, {-1, -2}, true, "t_iden", INT32, INT32, F32);

        // association between graph and features
        graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(graphFeatAssociation);
        loadDataset.addOutputData(featData);
        loadDataset.addOutputData(graphData);

        program.push_back(&loadDataset);
    }
    bool createdTransformedGraph = false;
    DataNode* graph;
    if (m1.data_transformations.size() > 0){ // need a transformed graph to be made
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], !m1.graph_transformations[UNWEIGHTED]);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            transformedGraphInfo->addOpt(COL_TILE_DOPT, std::to_string(m1.data_transformations[0].second));
        }
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
	    DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
	    DataNode* transformedGraph = new DataNode("graph_tile", graphData->getIType(), graphData->getNType(), graphData->getVType(), transformedRootGraphLevel);

        RelationEdge* trgraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(trgraphFeatAssociation);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
            /* tileTransformation->addParam(m1.data_transformations[0].second); */
            tileTransformation->addParam("65000"); // why is it a string parameter?
            TransformEdge* graphTrgraph = new TransformEdge(graphData, transformedGraph);
            graphTrgraph->addTransformation(tileTransformation);
            transforms.push_back(graphTrgraph);
        }

        graph = transformedGraph;
    }
    else{ // then make sure to modify the original graph with the schedule transformations!
        
        DataInfo* graphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        graphInfo->setDirected(!m1.graph_transformations[UNDIRECTED]);
        graphInfo->setWeighted(!m1.graph_transformations[UNWEIGHTED]);
        graph = graphData;
    }

    DataInfo* featInfo = dynamic_cast<DataInfo*>(featData->getData()->next());
    featInfo->setDims(-1, m1.graph_transformations[FEAT_SIZE]);
    
    TrainingLoopNode* trainingLoop = new TrainingLoopNode(m1.iterations, CROSS_ENTROPY, ADAM, m1.validation_step);
    DataNode* connectNode;
    for (int i = 0; i < m1.num_layers; i++){
        connectNode = addLayer(i, connectNode, graph, featData, trainingLoop); 
    }
    program.push_back(trainingLoop);

    cout << "IR Generated!\n";
}

void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}