%{
#include <iostream>
#include <string.h>
#include <vector>
#include <map>
// #include "../ir/data.h"
// #include "../ir/compute.h"
// #include "../ir/frontend_metadata.h"
#include "context.h"
using namespace std;

extern int yydebug;
extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);

extern ModelConfig m1; // just considering one model per input file for now
int debug = 0;

DataNode* normData; // very temp solution, find fix asap
DataNode* reluDataPrevLayer;

// extern vector<CIRNode*> programVec;
// extern vector<RelationEdge*> dependenciesVec;
// extern vector<RelationEdge*> associationsVec;
// extern vector<TransformEdge*> transformsVec;

bool operator_reordering = true;
bool sparse_rewrites = true;
bool training_subgraph = true;
bool train_code_motion = true;
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
%token<sval> AGGR_INIT FN_ARG MUL_SUM MUL_MEAN DSL_DOT FFN_OUT SIZE_FN 
%token<sval> GRAPH_ATTR FEAT_ATTR RELU LABEL_ATTR DEGREE_ATTR NODE_ATTR LEAKY_RELU
%token<sval> POW SCALAR_INIT IS_SPARSE
%token<sval> POW SCALAR_INIT
%token<sval> COLTILE FEAT_SIZE_ASSIGN LABEL_SIZE_ASSIGN COARSEN SRC_ATTR DST_ATTR;
%token<sval> INTEGER FLOAT SOFTMAX INIT_WEIGHT;
%token<sval> LBRACE RBRACE DOT COMMA;
%token<sval> TR FA OP_REORD SPARSE_REWRITES TRAIN_SUBGRAPH TRAIN_CODE_MOTION;
%token<sval> PLUS MINUS MULTIPLY DIVIDE;
%token<sval> FFN NULL_KEY EDGE_AGGR_INIT SUM EDGE_ATTR VAL_ATTR
%token<sval> LOSS OPTIMIZER RMSE_LOSS ADAM_T DSL_FN RELAXNLN QUANT RABBIT_REORDER_OP SAMPLE_RANDOM_OP AGGR LSQBRA RSQBRA IF ELSE DO WHILE NOT AND OR NOTEQ EQ GREATER LESS GREATEREQ LESSEQ DATASET NONLN SENSEI_OP INT NEW

%type <ival> bool op arg args
%type <ltype> function update_op gnn_op
%type <sval> train_arg string data_var 
%type <vval> load_dataset function_init statement
%type <vval> algorithm layers layer_def statements
%type <vval> program schedule train_args
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
statement : IDENTIFIER ASSIGN function_init SEMICOLON
    {
    }
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON
    {
    }
    | IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR DOT DEGREE_ATTR LPAREN RPAREN SEMICOLON
    {
        if (debug == 2) cout << "layer op - get degrees\n";
        m1.layer_operations.push_back(GET_DEGREES);

    }
    | feats_s ASSIGN gnn_op SEMICOLON
    {
        if ($3 == NON_LINEARITY){
            if (debug == 2) cout << "layer op - nonln\n";
            m1.layer_operations.push_back($3);
        }
    }
    | edge_vals ASSIGN gnn_op SEMICOLON
    {
        if ($3 == SOFTMAX_OP){
            if (debug == 2) cout << "layer op - softmax\n";
            m1.layer_operations.push_back($3);
        }
    }
    | IDENTIFIER ASSIGN gnn_op SEMICOLON
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
        else if ($3 == ATTEN_L){
            if (debug == 2) cout << "layer op - atten_l\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == ATTEN_R){
            if (debug == 2) cout << "layer op - atten_r\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == ATTN){
            if (debug == 2) cout << "layer op - attn\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == LEAKY_RELU_OP){
            if (debug == 2) cout << "layer op - leaky_relu\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == MULT_SCALAR_FEATS){
            if (debug == 2) cout << "layer op - mult scalar feats\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == ADD_SCALAR_AGGR){
            if (debug == 2) cout << "layer op - add scalar aggr\n";
            m1.layer_operations.push_back($3);
        }
        else if ($3 == SAGE_OPS){
            if (debug == 2) cout << "layer op - other sage ops\n";
            m1.layer_operations.insert(m1.layer_operations.begin(), GET_NORMALIZATION);
            m1.layer_operations.insert(m1.layer_operations.begin(), GET_DEGREES);
            m1.layer_operations.push_back(MULT_NORM_RES);
            // m1.layer_operations.push_back(FEED_FORWARD_NN);
            m1.layer_operations.push_back(ADD_TWO_FFN);
        }
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
layer_init : IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON { 
    m1.nonln_present.push_back(!$5);
    m1.num_layers++; 
}
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
        else if ($2 == 1){
            $$ = ADD_SCALAR_AGGR;
        }
    }
    | function 
    { $$ = $1; }
    | function op feats_s
    {
        $$ = $1;
    }
    | function op data_var
    {
        if ($1 == ATTEN_L || $1 == ATTEN_R){
            if (string($3) == "src_nodes"){
                $$ = ATTEN_L;
            }
            else if (string($3) == "dst_nodes"){
                $$ = ATTEN_R;
            }
        }
        // else if ($1 == MULT_SCALAR_FEATS){
        //     $$ = $1;
        // }
    }
;
function : IDENTIFIER LPAREN data_var COMMA data_var RPAREN
    {
        // for now we basically know this is an aggregate (but will need to update later)
        $$ = MESSAGE_PASSING_AGGREGATE;

    }
    | IDENTIFIER LPAREN data_var COMMA data_var COMMA data_var RPAREN
    {
        $$ = ATTN;
    }
    | IDENTIFIER LPAREN data_var RPAREN
    {
        // similarly know that this is relu
        $$ = NON_LINEARITY;
    }
    /* | DSL_DOT update_op op DSL_DOT update_op
    {
        $$ = SAGE_OPS;
    } */
    | DSL_DOT ffn_aggr op DSL_DOT ffn_aggr 
    {
        $$ = SAGE_OPS;
    }
    | DSL_DOT update_op
    {
        $$ = $2;
    }
;
ffn_aggr : FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN {};
update_op : FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN
    { $$ = FEED_FORWARD_NN; }
    | FFN LPAREN data_var COMMA FFN_OUT INTEGER RPAREN
    { 
        int n = m1.layer_operations.size();
        if (n > 0 && m1.layer_operations[n-1] != ATTEN_L)
            $$ = ATTEN_L; 
    }
    | RELU LPAREN data_var RPAREN
    { $$ = NON_LINEARITY; }
    | LEAKY_RELU LPAREN data_var op data_var RPAREN
    { $$ = LEAKY_RELU_OP; } 
    | POW LPAREN data_var COMMA FLOAT RPAREN
    {  m1.normalization_value = stof($5); $$ = GET_NORMALIZATION; }
    | POW LPAREN data_var COMMA INTEGER RPAREN
    { $$ = GET_NORMALIZATION; }
    | SCALAR_INIT LPAREN INTEGER RPAREN
    { $$ = MULT_SCALAR_FEATS; }
    | SOFTMAX LPAREN data_var COMMA data_var RPAREN
    { $$ = SOFTMAX_OP; }
    | INIT_WEIGHT LPAREN RPAREN
    { $$ = ATTEN_L; } // doesn't matter left or right will be updated anyway
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
    {  m1.addGraphTransformation(UNDIRECTED, (float) $7);  }
    | data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON
    {  m1.addGraphTransformation(UNWEIGHTED, (float) $7); }
    | FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON
    {  m1.addGraphTransformation(FEAT_SIZE, atof($3));  }
    | LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON 
    {  m1.addGraphTransformation(LABEL_SIZE, atof($3));  }
    | data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON 
    {  m1.addDataTransformation(COL_TILE, atof($7));  }
    | data_var ASSIGN data_var DOT IS_SPARSE LPAREN bool RPAREN SEMICOLON
    {  m1.addGraphTransformation(SPARSE, (float) $7); }
;
// this is where to do compute transform
function_transform : data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON
    {m1.addComputeTransformation(COARSE, atof($7)); }
    | OP_REORD LPAREN bool RPAREN SEMICOLON
    { operator_reordering = $3;}
    | SPARSE_REWRITES LPAREN bool RPAREN SEMICOLON
    { sparse_rewrites = $3; }
    | TRAIN_SUBGRAPH LPAREN bool RPAREN SEMICOLON
    { training_subgraph = $3; }
    | TRAIN_CODE_MOTION LPAREN bool RPAREN SEMICOLON
    { train_code_motion = $3; }
;
feats_s : IDENTIFIER DOT NODE_ATTR DOT FEAT_ATTR {};
edge_vals : IDENTIFIER DOT EDGE_ATTR DOT VAL_ATTR {};
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
    | data_var DOT EDGE_ATTR
    {
        $$ = strdup("edge_attr");
        free($1);
    }
    | data_var DOT VAL_ATTR
    {
        $$ = strdup("val_attr");
        free($1);
    }
;
function_init : AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN
    {}
    | EDGE_AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN
    {}
;
semiring_op : MUL_SUM
    {}
    | MUL_MEAN {}
    | SUM {}
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
args : { $$ = 0; }
    | args arg {
        if ($1 == 1 || $2 == 1){
            $$ = 1;
        }
        else{
            $$ = 0;
        }
    }
;
arg : INTEGER COMMA { m1.output_input_classes.push_back(atof($1)); } | INTEGER
    { m1.output_input_classes.push_back(atof($1)); }
    | NULL_KEY COMMA { $$ = 1;} | NULL_KEY {
        $$ = 1; // if arg == 1 then no nonln in this layer
    } // not matching if its same arg number as nonln, but for now its okay
    | data_var COMMA { } | data_var {}
    | DSL_DOT RELU { $$ = 0; } | DSL_DOT RELU COMMA  { $$ = 0; }
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
	GALAFEContext::associations.push_back(onesTensorGraphAssociation);

	// The actual degrees calculation
	ForwardNode* degreesOp = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_DIRECT);
	DataInfo* degreesInfo = new DataInfo(RM_DTYPE);
	degreesInfo->setDims(-1, 1);
	DataLevel* rootDegreesLevel = new DataLevel(degreesInfo, true);
	DataNode* degreesData = new DataNode("degrees", INT32, INT32, F32, rootDegreesLevel);
	degreesOp->addInputData(onesData);
	degreesOp->addInputData(graphData);
	degreesOp->addOutputData(degreesData);
	if (m1.compute_transformations.size() > 0)
	    degreesOp->addOpt(COARSE_COPT, m1.compute_transformations[0].second);
	trainingLoop->addLoopNode(degreesOp);
	RelationEdge* degreesOpOnesDependency = new RelationEdge(onesData, ALL_RELATION, degreesData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(degreesOpOnesDependency);
	RelationEdge* degreesOpGraphDependency = new RelationEdge(graphData, ALL_RELATION, degreesData, ROWS_RELATION);
	GALAFEContext::dependencies.push_back(degreesOpGraphDependency);
    return degreesData;
}
DataNode* addNormalization_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop){
    if (debug == 2) cout << "normalization setup\n";
    ForwardNode* powerOp = new ForwardNode(POINTWISE, POWER_OP);
	powerOp->addParam(to_string(m1.normalization_value)); 
    DataNode* normData = createDataNode(RM_DTYPE, false, false, {-1, 1}, true, "norm", INT32, INT32, F32);
	powerOp->addInputData(prevData);
	powerOp->addOutputData(normData);
	trainingLoop->addLoopNode(powerOp);
	RelationEdge* powerOpDegreesDependency = new RelationEdge(prevData, ALL_RELATION, normData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(powerOpDegreesDependency);
    return normData;
}
DataNode* addNormCalc_CIR(DataNode* normData, DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum, bool featInput, bool sage){ // prevData is either feat or res
    if (debug == 2) cout << "normalization-calculation\n";
	// 1st normalization calculation (or "Mean Calculation")
	ForwardNode* normFeat1 = new ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
    pair<int,int> normFeat1Data_inputDim = {-1, featInput ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes[layerNum]};
    string name = sage ? "res_n" : "res";
    DataNode* normFeat1Data = createDataNode(RM_DTYPE, false, false, normFeat1Data_inputDim, true, name, INT32, INT32, F32);
	normFeat1->addInputData(normData);
	normFeat1->addInputData(prevData);
	normFeat1->addOutputData(normFeat1Data);
	trainingLoop->addLoopNode(normFeat1);
	RelationEdge* normFeat1NormDependency = new RelationEdge(normData, ALL_RELATION, normFeat1Data, ROWS_RELATION);
	GALAFEContext::dependencies.push_back(normFeat1NormDependency);
	RelationEdge* normFeat1FeatDependency = new RelationEdge(prevData, ALL_RELATION, normFeat1Data, ALL_RELATION);
	GALAFEContext::dependencies.push_back(normFeat1FeatDependency);
	RelationEdge* normFeat1NormFeatAssociation = new RelationEdge(normData, ALL_RELATION, prevData, ROWS_RELATION);
	GALAFEContext::associations.push_back(normFeat1NormFeatAssociation);
    return normFeat1Data;

}
DataNode* addAggregate_CIR(DataNode* prevData, DataNode* graphData, TrainingLoopNode* trainingLoop, int layerNum, int gin, int sage){
    if (debug == 2) cout << "aggregate" << '\n';
    ForwardNode* aggregate = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    pair<int,int> outputData_inputDim = {-1, (layerNum == 0) ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes[layerNum]};
    DataNode* outputData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, gin || sage ? "res_n" : "res", INT32, INT32, F32);

    // // TODO Temp fix
    // aggregate->addOpt(COARSE_COPT, 2);
    
    aggregate->addInputData(prevData);
    aggregate->addInputData(graphData); 
    aggregate->addOutputData(outputData);
    if (m1.compute_transformations.size() > 0)
	    aggregate->addOpt(COARSE_COPT, m1.compute_transformations[0].second);
    trainingLoop->addLoopNode(aggregate);

    // Relation (dependency) between features and aggregated output
    RelationEdge* inOutAggrRelationFeat = new RelationEdge(prevData, ALL_RELATION, outputData, ALL_RELATION);
    RelationEdge* inOutAggrRelationGraph = new RelationEdge(graphData, ALL_RELATION, outputData, ALL_RELATION);
    GALAFEContext::dependencies.push_back(inOutAggrRelationFeat);
    GALAFEContext::dependencies.push_back(inOutAggrRelationGraph);
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
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes[layerNum]};
        resInputDim = {-1, m1.output_input_classes[layerNum]};
    }
    else{
        weightInputDim = {m1.output_input_classes[layerNum], m1.graph_transformations[LABEL_SIZE]};
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
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationFeat);
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationWeight);
    // Relation (association) between aggregate node and weight
    RelationEdge* inOutWeightAssociation = new RelationEdge(prevData, ROWS_RELATION, weightData, COLS_RELATION);
    GALAFEContext::associations.push_back(inOutWeightAssociation);
    return resData;

}
DataNode* addReLU_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "relu\n";
    // ReLU operation
	ForwardNode* reluOp = new ForwardNode(POINTWISE, NON_LNR_OP_RELU);
    pair<int,int> reluData_inputDim = {-1, (layerNum == 0) ? m1.output_input_classes[layerNum] : m1.graph_transformations[LABEL_SIZE]};
    DataNode* reluData = createDataNode(RM_DTYPE, false, false, reluData_inputDim, true, "res", INT32, INT32, F32);
	reluOp->addInputData(prevData);
	reluOp->addOutputData(reluData);
	trainingLoop->addLoopNode(reluOp);
	RelationEdge* reluOpOnesDependency = new RelationEdge(prevData, ALL_RELATION, reluData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(reluOpOnesDependency);
    return reluData;
}
DataNode* addAttentionWeight_L(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "add attention weight left\n";
    // Add attention weight operation (L side)
	// L side
	ForwardNode* atten_l = new ForwardNode(UPDATE_NODE, FFN_OP_EDGE);
	// Weight as a matrix in the DIR
	DataInfo* attenLWeightInfo = new DataInfo(CM_DTYPE);
    pair<int,int> weightInputDim;
    pair<int,int> resInputDim;
    if (layerNum == 0){
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes[layerNum]};
        resInputDim = {-1, m1.output_input_classes[layerNum]};
    }
    else{
        weightInputDim = {m1.output_input_classes[layerNum], m1.graph_transformations[LABEL_SIZE]};
        resInputDim = {-1, m1.graph_transformations[LABEL_SIZE]};
    }
	attenLWeightInfo->setDims(weightInputDim.first, weightInputDim.second); // -2=input embedding dimension, -3=output classes
    std::string attenLWeightName = "attenLWeight" + to_string(layerNum+1);
	DataLevel* attenLWeightLevel = new DataLevel(attenLWeightInfo, true);
	DataNode* attenLWeightData = new DataNode(attenLWeightName, INT32, INT32, F32, attenLWeightLevel);
	// Res DIR
    std::string attenLDataName;
    if (layerNum != 0)
        attenLDataName = "attenL_" + to_string(layerNum+1);
    else
        attenLDataName = "attenL";
	DataInfo* attenLInfo = new DataInfo(RM_DTYPE);
	attenLInfo->setDims(resInputDim.first, resInputDim.second); // -1=N=232965, the number of nodes in the graph, -3=output classes
	DataLevel* rootAttenLLevel = new DataLevel(attenLInfo, true);
	DataNode* attenLData = new DataNode(attenLDataName, INT32, INT32, F32, rootAttenLLevel);
	// set dimenions from the new schedule information
	atten_l->addInputData(prevData);
	atten_l->addInputData(attenLWeightData);
	atten_l->addOutputData(attenLData);
	trainingLoop->addLoopNode(atten_l);
	//* Dependencies
	RelationEdge* inOutAttenLtDepRelationFeat = new RelationEdge(prevData, ALL_RELATION, attenLData, ALL_RELATION);
	RelationEdge* inOutAttenLDepRelationWeight = new RelationEdge(attenLWeightData, COLS_RELATION, attenLData, ROWS_RELATION);
	GALAFEContext::dependencies.push_back(inOutAttenLtDepRelationFeat);
	GALAFEContext::dependencies.push_back(inOutAttenLDepRelationWeight);
	RelationEdge* inOutAttenLAssociation = new RelationEdge(prevData, ROWS_RELATION, attenLWeightData, COLS_RELATION);
	GALAFEContext::associations.push_back(inOutAttenLAssociation);
    return attenLData;
}
DataNode* addAttentionWeight_R(DataNode* prevData, DataNode* resData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "add attention weight right\n";
    // R side
	ForwardNode* atten_r = new ForwardNode(UPDATE_NODE, FFN_OP_EDGE);
    pair<int,int> weightInputDim;
    pair<int,int> resInputDim;
	DataInfo* attenRWeightInfo = new DataInfo(CM_DTYPE);
    if (layerNum == 0){
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes[layerNum]};
        resInputDim = {-1, m1.output_input_classes[layerNum]};
    }
    else{
        weightInputDim = {m1.output_input_classes[layerNum], m1.graph_transformations[LABEL_SIZE]};
        resInputDim = {-1, m1.graph_transformations[LABEL_SIZE]};
    }
    std::string attenLWeightName = "attenRWeight" + to_string(layerNum+1);
	attenRWeightInfo->setDims(weightInputDim.first, weightInputDim.second); // -2=input embedding dimension, -3=output classes
	DataLevel* attenRWeightLevel = new DataLevel(attenRWeightInfo, true);
	DataNode* attenRWeightData = new DataNode(attenLWeightName, INT32, INT32, F32, attenRWeightLevel);
	// Res DIR
    std::string attenRDataName;
    if (layerNum != 0)
        attenRDataName = "attenR_" + to_string(layerNum+1);
    else
        attenRDataName = "attenR";
	DataInfo* attenRInfo = new DataInfo(RM_DTYPE);
	attenRInfo->setDims(resInputDim.first, resInputDim.second); // -1=N=232965, the number of nodes in the graph, -3=output classes
	DataLevel* rootAttenRLevel = new DataLevel(attenRInfo, true);
	DataNode* attenRData = new DataNode(attenRDataName, INT32, INT32, F32, rootAttenRLevel);

	atten_r->addInputData(resData);
	atten_r->addInputData(attenRWeightData);
	atten_r->addOutputData(attenRData);
	trainingLoop->addLoopNode(atten_r);
	//* Dependencies
	RelationEdge* inOutAttenRtDepRelationFeat = new RelationEdge(resData, ALL_RELATION, attenRData, ALL_RELATION);
	RelationEdge* inOutAttenRDepRelationWeight = new RelationEdge(attenRWeightData, COLS_RELATION, attenRData, ROWS_RELATION);
	GALAFEContext::dependencies.push_back(inOutAttenRtDepRelationFeat);
	GALAFEContext::dependencies.push_back(inOutAttenRDepRelationWeight);
	RelationEdge* inOutAttenRAssociation = new RelationEdge(resData, ROWS_RELATION, attenRWeightData, COLS_RELATION);
	GALAFEContext::associations.push_back(inOutAttenRAssociation);
    return attenRData;
}
DataNode* addAttn(DataNode* attenLData, DataNode* attenRData, DataNode* graphData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "general attention\n";
	// Edge aggregation
	auto aggregateEdge = new ForwardNode(AGGREGATE_EDGE, AGGREGATE_EDGE_SUM_OP);
	auto aggrEdgeInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], true);
	aggrEdgeInfo->addOpt(COL_TILE_DOPT, m1.data_transformations[0].second);
	aggrEdgeInfo->setIndex(0);
	aggrEdgeInfo->setDerived(true);
	// aggrEdgeInfo.setDims(-4, 1); //-4=E=114M (E = Edges)
	auto rootAggrEdgeLevel = new DataLevel(aggrEdgeInfo, true);
	auto aggrEdgeData = new DataNode("attn", INT32, INT32, F32, rootAggrEdgeLevel);
	aggregateEdge->addInputData(attenLData);
	aggregateEdge->addInputData(attenRData);
	aggregateEdge->addInputData(graphData);
	aggregateEdge->addOutputData(aggrEdgeData);
	// TODO add optimizations
	trainingLoop->addLoopNode(aggregateEdge);
	//* Dependencies
	// Dependency relation between the features and the aggregated output
	auto inOutEdgeAggrLRelationFeat = new RelationEdge(attenLData, ALL_RELATION, aggrEdgeData, ROWS_RELATION);
	auto inOutEdgeAggrRRelationFeat = new RelationEdge(attenRData, ALL_RELATION, aggrEdgeData, COLS_RELATION);
	auto inOutEdgeAggrRelationGraph = new RelationEdge(graphData, ALL_RELATION, aggrEdgeData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(inOutEdgeAggrLRelationFeat);
	GALAFEContext::dependencies.push_back(inOutEdgeAggrRRelationFeat);
	GALAFEContext::dependencies.push_back(inOutEdgeAggrRelationGraph);
	auto graphEdgeAggrLAssociation = new RelationEdge(graphData, ROWS_RELATION, attenLData, ALL_RELATION);
	auto graphEdgeAggrRAssociation = new RelationEdge(graphData, COLS_RELATION, attenRData, ALL_RELATION);
	GALAFEContext::associations.push_back(graphEdgeAggrLAssociation);
	GALAFEContext::associations.push_back(graphEdgeAggrRAssociation);
    return aggrEdgeData;
}
DataNode* addSoftmax_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "softmax\n";
    ForwardNode* softmaxOp = new ForwardNode(UPDATE_EDGE, NON_LNR_OP_SOFTMAX);
	DataInfo* softmaxInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], true);
    softmaxInfo->addOpt(COL_TILE_DOPT, m1.data_transformations[0].second);
	softmaxInfo->setIndex(0);
	softmaxInfo->setDerived(true);
	// leakyReluInfo.setDims(-4, 1);
	// TODO Temp fix
	prevData->getDataInfo()->addOpt(COL_TILE_DOPT, "300000");
	DataLevel* rootSoftmaxLevel = new DataLevel(softmaxInfo, true);
	DataNode* softmaxData = new DataNode("attn", INT32, INT32, F32, rootSoftmaxLevel);
	softmaxOp->addInputData(prevData);
	softmaxOp->addOutputData(softmaxData);
	trainingLoop->addLoopNode(softmaxOp);
	RelationEdge* softmaxOpOnesDependency = new RelationEdge(prevData, ALL_RELATION, softmaxData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(softmaxOpOnesDependency);
    return softmaxData;
}
DataNode* addLeakyReLU(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "leakyRelu\n"; 
	// Leaky ReLU operation
	ForwardNode* leakyReluOp = new ForwardNode(UPDATE_EDGE, NON_LNR_OP_LEAKY_RELU);
	leakyReluOp->addParam("0.2"); // TODO: avoid hardcoding
	DataInfo* leakyReluInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], true);
	// leakyReluInfo.setDims(-4, 1);
	DataLevel* rootLeakyReluLevel = new DataLevel(leakyReluInfo, true);
	DataNode* leakyReluData = new DataNode("attn", INT32, INT32, F32, rootLeakyReluLevel);
	leakyReluOp->addInputData(prevData);
	leakyReluOp->addOutputData(leakyReluData);
	trainingLoop->addLoopNode(leakyReluOp);
	RelationEdge* leakyReluOpOnesDependency = new RelationEdge(prevData, ALL_RELATION, leakyReluData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(leakyReluOpOnesDependency);
    return leakyReluData;
}
DataNode* add_mulScalarEPS_CIR(DataNode* featData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "mul-scalar-eps\n";
    // Scalar multiply res
	ForwardNode* scalarEps = new ForwardNode(POINTWISE, SCALAR_ADD_EPS_MULTIPLY_OP);
	scalarEps->addParam("1"); // TODO: change this to user input instead of hardcode
    pair<int,int> outputData_inputDim = {-1, (layerNum == 0) ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes[layerNum]};
    DataNode* scalarEpsData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, "res", INT32, INT32, F32);
	scalarEps->addInputData(featData);
	scalarEps->addOutputData(scalarEpsData);
	trainingLoop->addLoopNode(scalarEps);
	RelationEdge* scalarEpsDependency = new RelationEdge(featData, ALL_RELATION, scalarEpsData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(scalarEpsDependency);
    return scalarEpsData;
}
DataNode* add_addScalarFeats_CIR(DataNode* prevData, DataNode* aggrOutput, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "add-scalar-eps\n";
    // Add epsilon mult and scalar mults
	ForwardNode* normFeat = new ForwardNode(UPDATE_NODE, ADD_OP);
    /* DataNode* scalarEpsData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, "res", INT32, INT32, F32); */
    pair<int,int> outputData_inputDim = {-1, (layerNum == 0) ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes[layerNum]};
    DataNode* normFeatData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, "res", INT32, INT32, F32);
	normFeat->addInputData(prevData);
	normFeat->addInputData(aggrOutput);
	normFeat->addOutputData(normFeatData);
	trainingLoop->addLoopNode(normFeat);
	RelationEdge* normFeatNormDependency = new RelationEdge(prevData, ALL_RELATION, normFeatData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(normFeatNormDependency);
	RelationEdge* normFeatFeatDependency = new RelationEdge(aggrOutput, ALL_RELATION, normFeatData, ALL_RELATION);
	GALAFEContext::dependencies.push_back(normFeatFeatDependency);
	RelationEdge* normFeatNormFeatAssociation = new RelationEdge(prevData, ALL_RELATION, aggrOutput, ALL_RELATION);
	GALAFEContext::associations.push_back(normFeatNormFeatAssociation);
    return normFeatData;
}
DataNode* add_addTwoFFN_CIR(DataNode* prevData, DataNode* featData, TrainingLoopNode* trainingLoop, int layerNum) {
    if (debug == 2) cout << "double ffn and add (only for sage)\n";
    // Add weight operation (res_n)
    auto* ffn = new ForwardNode(UPDATE_NODE, FFN_OP);
    
    pair<int,int> weightInputDim;
    pair<int,int> resInputDim;
    if (layerNum == 0){
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes[layerNum]};
        resInputDim = {-1, m1.output_input_classes[layerNum]};
    }
    else{
        weightInputDim = {m1.output_input_classes[layerNum], m1.graph_transformations[LABEL_SIZE]};
        resInputDim = {-1, m1.graph_transformations[LABEL_SIZE]};
    }
    // Weight as a matrix in the DIR
    auto* weightInfo = new DataInfo(CM_DTYPE);
    weightInfo->setDims(weightInputDim.first, weightInputDim.second); // Use model config for hidden dimension
    auto* weightLevel = new DataLevel(weightInfo, true);
    auto* weightData = new DataNode("weight1", INT32, INT32, F32, weightLevel);
    
    // Res DIR
    auto* resInfo = new DataInfo(RM_DTYPE);
    resInfo->setDims(resInputDim.first, resInputDim.second); // Use model config
    auto* rootResLevel = new DataLevel(resInfo, true);
    auto* resData = new DataNode("res_n", INT32, INT32, F32, rootResLevel);
    
    ffn->addInputData(prevData); // Use prevData instead of normFeatData
    ffn->addInputData(weightData);
    ffn->addOutputData(resData);
    trainingLoop->addLoopNode(ffn);
    
    // Dependencies
    auto* inOutWeightDepRelationFeat = new RelationEdge(prevData, ALL_RELATION, resData, ALL_RELATION);
    auto* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationFeat);
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationWeight);
    auto* inOutWeightAssociation = new RelationEdge(prevData, ROWS_RELATION, weightData, COLS_RELATION);
    GALAFEContext::associations.push_back(inOutWeightAssociation);
    
    // Add weight operation (res)
    auto* ffn2 = new ForwardNode(UPDATE_NODE, FFN_OP_SELF);
    
    // Weight as a matrix in the DIR
    auto* weightInfo2 = new DataInfo(CM_DTYPE);
    weightInfo2->setDims(weightInputDim.first, weightInputDim.second); // Use model config
    auto* weightLevel2 = new DataLevel(weightInfo2, true);
    auto* weightData2 = new DataNode("weight2", INT32, INT32, F32, weightLevel2);
    
    // Res DIR
    auto* resInfo2 = new DataInfo(RM_DTYPE);
    resInfo2->setDims(resInputDim.first, resInputDim.second); // Use model config
    auto* rootResLevel2 = new DataLevel(resInfo2, true);
    auto* resData2 = new DataNode("res", INT32, INT32, F32, rootResLevel2);
    
    // Set dimensions from the model config
    /* weightInfo2->setDims(m1->inputDim, m1->hiddenDim); // Use model config dimensions
    resInfo2->setDims(-1, m1->hiddenDim); // -1 for dynamic node count */
    
    ffn2->addInputData(prevData); // Use prevData instead of featData
    ffn2->addInputData(weightData2);
    ffn2->addOutputData(resData2);
    trainingLoop->addLoopNode(ffn2);
    
    // Dependencies
    auto* inOutWeightDepRelationFeat2 = new RelationEdge(featData, ALL_RELATION, resData2, ALL_RELATION);
    auto* inOutWeightDepRelationWeight2 = new RelationEdge(weightData2, COLS_RELATION, resData2, ROWS_RELATION);
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationFeat2);
    GALAFEContext::dependencies.push_back(inOutWeightDepRelationWeight2);
    auto* inOutWeightAssociation2 = new RelationEdge(prevData, ROWS_RELATION, weightData2, COLS_RELATION);
    GALAFEContext::associations.push_back(inOutWeightAssociation2);
    
    // Add weight updated res and res_n
    auto* addFeat = new ForwardNode(UPDATE_NODE, ADD_OP);
    auto* addFeatInfo = new DataInfo(RM_DTYPE);
    addFeatInfo->setDims(resInputDim.first, resInputDim.second); // Use model config
    auto* rootAddFeatLevel = new DataLevel(addFeatInfo, true);
    auto* addFeatData = new DataNode("res", INT32, INT32, F32, rootAddFeatLevel);
    
    addFeat->addInputData(resData);
    addFeat->addInputData(resData2);
    addFeat->addOutputData(addFeatData);
    trainingLoop->addLoopNode(addFeat);
    
    auto* resAddFeatDependency = new RelationEdge(resData, ALL_RELATION, addFeatData, ALL_RELATION);
    GALAFEContext::dependencies.push_back(resAddFeatDependency);
    auto* res2AddFeatDependency = new RelationEdge(resData2, ALL_RELATION, addFeatData, ALL_RELATION);
    GALAFEContext::dependencies.push_back(res2AddFeatDependency);
    auto* resRes2Dependency = new RelationEdge(resData, ALL_RELATION, resData2, ALL_RELATION);
    GALAFEContext::associations.push_back(resRes2Dependency);
    
    return addFeatData; // Return the final output data node
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
    DataNode* resData = NULL;
    DataNode* attenLData = NULL;
    DataNode* attenRData = NULL;
    DataNode* aggrData = NULL;
    for (int i = 0; i < m1.layer_operations.size(); i++){
        LayerOpType t = m1.layer_operations[i];
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
                if (m1.layer_operations.size() > 6 && trainingLoop->getLoopNodeNum() < 5) // temporary fix to see if use featData or resData
                    prevData = addNormCalc_CIR(normData, featData, trainingLoop, layerNum, true, i > 0 && m1.layer_operations[i-1] == MESSAGE_PASSING_AGGREGATE);
                else
                    prevData = addNormCalc_CIR(normData, prevData, trainingLoop, layerNum, false, i > 0 && m1.layer_operations[i-1] == MESSAGE_PASSING_AGGREGATE);
                break;
            case MESSAGE_PASSING_AGGREGATE:
                if (i > 0 && m1.layer_operations[i-1] == SOFTMAX_OP)
                    prevData = addAggregate_CIR(resData, prevData, trainingLoop, layerNum, 0, i < m1.layer_operations.size() && m1.layer_operations[i+1] == MULT_NORM_RES);
                else if (i > 0 && m1.layer_operations[i-1] == GET_NORMALIZATION && trainingLoop->getLoopNodeNum() <= 3)
                    prevData = addAggregate_CIR(featData, graphData, trainingLoop, layerNum, 0, i < m1.layer_operations.size() && m1.layer_operations[i+1] == MULT_NORM_RES);
                else
                    if (i < m1.layer_operations.size() && (m1.layer_operations[i+1] == MULT_SCALAR_FEATS))
                        prevData = addAggregate_CIR(prevData, graphData, trainingLoop, layerNum, 1, i < m1.layer_operations.size() && m1.layer_operations[i+1] == MULT_NORM_RES);
                    else
                        prevData = addAggregate_CIR(prevData, graphData, trainingLoop, layerNum, 0,  i < m1.layer_operations.size() && m1.layer_operations[i+1] == MULT_NORM_RES);
                aggrData = prevData;
                break;
            case FEED_FORWARD_NN:
                prevData = addFFN_CIR(prevData, trainingLoop, layerNum);
                resData = prevData;
                break;
            case NON_LINEARITY:
                if (m1.nonln_present[layerNum]){
                    prevData = addReLU_CIR(prevData, trainingLoop, layerNum);
                    reluDataPrevLayer = prevData;
                }
                break;
            case ATTEN_L:
                prevData = addAttentionWeight_L(prevData, trainingLoop, layerNum);
                attenLData = prevData; prevData = addAttentionWeight_R(prevData, resData, trainingLoop, layerNum);
                attenRData = prevData;
                if (attenLData && attenRData){
                    prevData = addAttn(attenLData, attenRData, graphData, trainingLoop, layerNum);
                }
                break;
            case ATTEN_R:
                /* prevData = addAttentionWeight_R(prevData, resData, trainingLoop, layerNum);
                attenRData = prevData;
                if (attenLData && attenRData){
                    prevData = addAttn(attenLData, attenRData, graphData, trainingLoop, layerNum);
                } */
                break;
            case ATTN:
                prevData = addLeakyReLU(prevData, trainingLoop, layerNum);
                /* prevData = addAttn(attenLData, attenRData, graphData, trainingLoop, layerNum); */
                break;
            case SOFTMAX_OP:
                prevData = addSoftmax_CIR(prevData, trainingLoop, layerNum);
                break;
            case LEAKY_RELU_OP:
                prevData = addLeakyReLU(prevData, trainingLoop, layerNum);
                break;
            case MULT_SCALAR_FEATS:
                if (trainingLoop->getLoopNodeNum() <= 5)
                    prevData = add_mulScalarEPS_CIR(featData, trainingLoop, layerNum);
                else
                    prevData = add_mulScalarEPS_CIR(reluDataPrevLayer, trainingLoop, layerNum);
                break;
            case ADD_SCALAR_AGGR:
                prevData = add_addScalarFeats_CIR(prevData, aggrData, trainingLoop, layerNum);
                break;
            case ADD_TWO_FFN:
                prevData = add_addTwoFFN_CIR(prevData, featData, trainingLoop, layerNum);
                break;
            default:
                cout << "UNKNOWN LAYER OP TYPE\n";
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
        auto loadDataset = new ForwardNode(POINTWISE, LOAD_OP);
        loadDataset->addParam(m1.dataset_name);
        // TODO Temp fix
        graphData = createDataNode(CSR_STYPE, false, false, {0,0}, true, "adj0", INT32, INT32, F32);
        featData = createDataNode(RM_DTYPE, false, false, {-1, -2}, true, "t_iden", INT32, INT32, F32);

        // association between graph and features
        graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        GALAFEContext::associations.push_back(graphFeatAssociation);
        loadDataset->addOutputData(featData);
        loadDataset->addOutputData(graphData);

        GALAFEContext::program.push_back(loadDataset);
    }
    bool createdTransformedGraph = false;
    DataNode* graph;
    if (m1.data_transformations.size() > 0){ // need a transformed graph to be made
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], !m1.graph_transformations[UNWEIGHTED]);
        transformedGraphInfo->setSparse(m1.graph_transformations[SPARSE]);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            transformedGraphInfo->addOpt(COL_TILE_DOPT, to_string(m1.data_transformations[0].second));
        }
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
	    DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
	    DataNode* transformedGraph = new DataNode("graph_tile", graphData->getIType(), graphData->getNType(), graphData->getVType(), transformedRootGraphLevel);

        RelationEdge* trgraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, featData, ROWS_RELATION);
        GALAFEContext::associations.push_back(trgraphFeatAssociation);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
            tileTransformation->addParam(to_string(m1.data_transformations[0].second));
            /* tileTransformation->addParam("65000"); // why is it a string parameter? */
            TransformEdge* graphTrgraph = new TransformEdge(graphData, transformedGraph);
            graphTrgraph->addTransformation(tileTransformation);
            GALAFEContext::transforms.push_back(graphTrgraph);
        }

        graph = transformedGraph;
    }
    else{ // then make sure to modify the original graph with the schedule transformations!
        
        DataInfo* graphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        graphInfo->setDirected(!m1.graph_transformations[UNDIRECTED]);
        graphInfo->setWeighted(!m1.graph_transformations[UNWEIGHTED]);
        graphInfo->setSparse(m1.graph_transformations[SPARSE]);
        graph = graphData;
    }

    DataInfo* featInfo = dynamic_cast<DataInfo*>(featData->getData()->next());
    featInfo->setDims(-1, m1.graph_transformations[FEAT_SIZE]);
    
    TrainingLoopNode* trainingLoop = new TrainingLoopNode(m1.iterations, CROSS_ENTROPY, ADAM, m1.validation_step);
    DataNode* connectNode = featData;
    for (int i = 0; i < m1.num_layers; i++){
        connectNode = addLayer(i, connectNode, graph, featData, trainingLoop); 
    }
    GALAFEContext::program.push_back(trainingLoop);

    cout << "IR Generated!\n";
}

void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}