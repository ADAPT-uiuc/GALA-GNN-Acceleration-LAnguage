/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "frontend.y"

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
int debug = 0;
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
    if (debug) cout << "aggregate" << '\n';
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
    if (debug) cout << "ffn" << '\n';
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
    if (debug) cout << "normalization setup\n";
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
    if (debug) cout << "normalization-calculation\n";
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
    if (debug) cout << "relu\n";
    // ReLU operation
	ForwardNode* reluOp = new ForwardNode(POINTWISE, NON_LNR_OP_RELU);
	DataInfo* reluInfo = new DataInfo(RM_DTYPE);
	reluInfo->setDims(-1, 32);
	DataLevel* rootReluLevel = new DataLevel(reluInfo, true);
	DataNode* reluData = new DataNode("res-relu", INT32, INT32, F32, rootReluLevel);
	reluOp->addInputData(defaultInput);
	reluOp->addOutputData(reluData);
    dataNodeMap["reluData"] = reluData;
	mainTrainingLoop->addLoopNode(reluOp);
	RelationEdge* reluOpOnesDependency = new RelationEdge(defaultInput, ALL_RELATION, reluData, ALL_RELATION);
	dependencies.push_back(reluOpOnesDependency);
}
void addDegrees_CIR(DataNode* defaultInput, DataNode* graph){
    if (debug) cout << "degrees\n";
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
void mulScalarEPS_CIR(DataNode* featData){
    if (debug) cout << "mul-scalar-eps\n";
    // Scalar multiply res
	ForwardNode* scalarEps = new ForwardNode(POINTWISE, SCALAR_ADD_EPS_MULTIPLY_OP);
	scalarEps->addParam("1"); // TODO: change this to user input instead of hardcode
	DataInfo* scalarInfo = new DataInfo(RM_DTYPE);
	scalarInfo->setDims(-1, -2);
	DataLevel* rootScalarEpsLevel = new DataLevel(scalarInfo, true);
	DataNode* scalarEpsData = new DataNode("res-mulScalarEPS", INT32, INT32, F32, rootScalarEpsLevel);
	scalarEps->addInputData(featData);
	scalarEps->addOutputData(scalarEpsData);
	mainTrainingLoop->addLoopNode(scalarEps);
	RelationEdge* scalarEpsDependency = new RelationEdge(featData, ALL_RELATION, scalarEpsData, ALL_RELATION);
	dependencies.push_back(scalarEpsDependency);

}
void addScalarEPS_CIR(DataNode* defaultInput, DataNode* aggrOutput){
    if (debug) cout << "add-scalar-eps\n";
    // Add epsilon mult and scalar mults
	ForwardNode* normFeat = new ForwardNode(UPDATE_NODE, ADD_OP);
	DataInfo* normFeatInfo = new DataInfo(RM_DTYPE);
	normFeatInfo->setDims(-1, -2);
	DataLevel* rootNormFeatLevel = new DataLevel(normFeatInfo, true);
	DataNode* normFeatData = new DataNode("res-addScalarEPS", INT32, INT32, F32, rootNormFeatLevel);
	normFeat->addInputData(defaultInput);
	normFeat->addInputData(aggrOutput);
	normFeat->addOutputData(normFeatData);
	mainTrainingLoop->addLoopNode(normFeat);
	RelationEdge* normFeatNormDependency = new RelationEdge(defaultInput, ALL_RELATION, normFeatData, ALL_RELATION);
	dependencies.push_back(normFeatNormDependency);
	RelationEdge* normFeatFeatDependency = new RelationEdge(aggrOutput, ALL_RELATION, normFeatData, ALL_RELATION);
	dependencies.push_back(normFeatFeatDependency);
	RelationEdge* normFeatNormFeatAssociation = new RelationEdge(defaultInput, ALL_RELATION, aggrOutput, ALL_RELATION);
	associations.push_back(normFeatNormFeatAssociation);
}
void addAttentionWeight_L(DataNode* resData){
    if (debug) cout << "add attention weight left\n";
    // Add attention weight operation (L side)
	// L side
	ForwardNode* atten_l = new ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	DataInfo* attenLWeightInfo = new DataInfo(CM_DTYPE);
	attenLWeightInfo->setDims(-2, 1); // -2=input embedding dimension, -3=output classes
	DataLevel* attenLWeightLevel = new DataLevel(attenLWeightInfo, true);
	DataNode* attenLWeightData = new DataNode("attenLWeight1", INT32, INT32, F32, attenLWeightLevel);
	// Res DIR
	DataInfo* attenLInfo = new DataInfo(RM_DTYPE);
	attenLInfo->setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	DataLevel* rootAttenLLevel = new DataLevel(attenLInfo, true);
	DataNode* attenLData = new DataNode("attenL", INT32, INT32, F32, rootAttenLLevel);
    dataNodeMap["attn-l"] = attenLWeightData;
	// set dimenions from the new schedule information
	atten_l->addInputData(resData);
	atten_l->addInputData(attenLWeightData);
	atten_l->addOutputData(attenLData);
	mainTrainingLoop->addLoopNode(atten_l);
	//* Dependencies
	RelationEdge* inOutAttenLtDepRelationFeat = new RelationEdge(resData, ALL_RELATION, attenLData, ALL_RELATION);
	RelationEdge* inOutAttenLDepRelationWeight = new RelationEdge(attenLWeightData, COLS_RELATION, attenLData, ROWS_RELATION);
	dependencies.push_back(inOutAttenLtDepRelationFeat);
	dependencies.push_back(inOutAttenLDepRelationWeight);
	RelationEdge* inOutAttenLAssociation = new RelationEdge(resData, ROWS_RELATION, attenLWeightData, COLS_RELATION);
	associations.push_back(inOutAttenLAssociation);
}
void addAttentionWeight_R(DataNode* resData){
    if (debug) cout << "add attention weight right\n";
// R side
	ForwardNode* atten_r = new ForwardNode(UPDATE_NODE, FFN_OP);
	// Weight as a matrix in the DIR
	DataInfo* attenRWeightInfo = new DataInfo(CM_DTYPE);
	attenRWeightInfo->setDims(-2, 1); // -2=input embedding dimension, -3=output classes
	DataLevel* attenRWeightLevel = new DataLevel(attenRWeightInfo, true);
	DataNode* attenRWeightData = new DataNode("attenRWeight1", INT32, INT32, F32, attenRWeightLevel);
	// Res DIR
	DataInfo* attenRInfo = new DataInfo(RM_DTYPE);
	attenRInfo->setDims(-1, 1); // -1=N=232965, the number of nodes in the graph, -3=output classes
	DataLevel* rootAttenRLevel = new DataLevel(attenRInfo, true);
	DataNode* attenRData = new DataNode("attenR", INT32, INT32, F32, rootAttenRLevel);
    dataNodeMap["attn-r"] = attenRWeightData;
	// set dimenions from the new schedule information
	atten_r->addInputData(resData);
	atten_r->addInputData(attenRWeightData);
	atten_r->addOutputData(attenRData);
	mainTrainingLoop->addLoopNode(atten_r);
	//* Dependencies
	RelationEdge* inOutAttenRtDepRelationFeat = new RelationEdge(resData, ALL_RELATION, attenRData, ALL_RELATION);
	RelationEdge* inOutAttenRDepRelationWeight = new RelationEdge(attenRWeightData, COLS_RELATION, attenRData, ROWS_RELATION);
	dependencies.push_back(inOutAttenRtDepRelationFeat);
	dependencies.push_back(inOutAttenRDepRelationWeight);
	RelationEdge* inOutAttenRAssociation = new RelationEdge(resData, ROWS_RELATION, attenRWeightData, COLS_RELATION);
	associations.push_back(inOutAttenRAssociation);
}
void addEdgeAggregation_CIR(DataNode* attn_l, DataNode* attn_r, DataNode* graph){
    cout << "edge aggregation\n";
    ForwardNode* aggregateEdge = new ForwardNode(AGGREGATE_EDGE, AGGREGATE_MUL_SUM_OP);
    DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graph->getData()->next());
	DataInfo* aggrEdgeInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), true);
	// aggrEdgeInfo.setDims(-4, 1); //-4=E=114M (E = Edges)
	DataLevel* rootAggrEdgeLevel = new DataLevel(aggrEdgeInfo, true);
	DataNode* aggrEdgeData = new DataNode("res-edgeAggr", INT32, INT32, F32, rootAggrEdgeLevel);
	aggregateEdge->addInputData(attn_l);
	aggregateEdge->addInputData(attn_r);
	aggregateEdge->addInputData(graph);
	aggregateEdge->addOutputData(aggrEdgeData);
	// TODO add optimizations
	mainTrainingLoop->addLoopNode(aggregateEdge);
}
void addLeakyReLU(DataNode* defaultInput, DataNode* graph){
    cout << "leakyRelu\n"; 
	// Leaky ReLU operation
	ForwardNode* leakyReluOp = new ForwardNode(UPDATE_EDGE, NON_LNR_OP_LEAKY_RELU);
	leakyReluOp->addParam("0.2"); // TODO: avoid hardcoding
    DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graph->getData()->next());
	DataInfo* leakyReluInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	DataLevel* rootLeakyReluLevel = new DataLevel(leakyReluInfo, true);
	DataNode* leakyReluData = new DataNode("res-leakyReLU", INT32, INT32, F32, rootLeakyReluLevel);
	leakyReluOp->addInputData(defaultInput);
	leakyReluOp->addOutputData(leakyReluData);
	mainTrainingLoop->addLoopNode(leakyReluOp);
	RelationEdge* leakyReluOpOnesDependency = new RelationEdge(defaultInput, ALL_RELATION, leakyReluData, ALL_RELATION);
	dependencies.push_back(leakyReluOpOnesDependency);
}
void addSoftmax_CIR(DataNode* defaultInput, DataNode* graph){
    cout << "softmax\n";
    ForwardNode* softmaxOp = new ForwardNode(UPDATE_EDGE, NON_LNR_OP_SOFTMAX);
    DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graph->getData()->next());
	DataInfo* softmaxInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), true);
	// leakyReluInfo.setDims(-4, 1);
	DataLevel* rootSoftmaxLevel = new DataLevel(softmaxInfo, true);
	DataNode* softmaxData = new DataNode("ses-softmax", INT32, INT32, F32, rootSoftmaxLevel);
	softmaxOp->addInputData(defaultInput);
	softmaxOp->addOutputData(softmaxData);
	mainTrainingLoop->addLoopNode(softmaxOp);
	RelationEdge* softmaxOpOnesDependency = new RelationEdge(defaultInput, ALL_RELATION, softmaxData, ALL_RELATION);
	dependencies.push_back(softmaxOpOnesDependency);
}

#line 376 "build/frontend.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "frontend.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_IDENTIFIER = 3,                 /* IDENTIFIER  */
  YYSYMBOL_ASSIGN = 4,                     /* ASSIGN  */
  YYSYMBOL_LOAD = 5,                       /* LOAD  */
  YYSYMBOL_LPAREN = 6,                     /* LPAREN  */
  YYSYMBOL_RPAREN = 7,                     /* RPAREN  */
  YYSYMBOL_SEMICOLON = 8,                  /* SEMICOLON  */
  YYSYMBOL_QUOTE = 9,                      /* QUOTE  */
  YYSYMBOL_COMMENT = 10,                   /* COMMENT  */
  YYSYMBOL_SET_UNWEIGHTED = 11,            /* SET_UNWEIGHTED  */
  YYSYMBOL_SET_UNDIRECTED = 12,            /* SET_UNDIRECTED  */
  YYSYMBOL_MODEL_W = 13,                   /* MODEL_W  */
  YYSYMBOL_EVAL = 14,                      /* EVAL  */
  YYSYMBOL_TRAIN = 15,                     /* TRAIN  */
  YYSYMBOL_LAYER = 16,                     /* LAYER  */
  YYSYMBOL_LOSS = 17,                      /* LOSS  */
  YYSYMBOL_OPTIMIZER = 18,                 /* OPTIMIZER  */
  YYSYMBOL_ITERS = 19,                     /* ITERS  */
  YYSYMBOL_VAL_STEP = 20,                  /* VAL_STEP  */
  YYSYMBOL_RMSE_LOSS = 21,                 /* RMSE_LOSS  */
  YYSYMBOL_ADAM_T = 22,                    /* ADAM_T  */
  YYSYMBOL_AGGR_INIT = 23,                 /* AGGR_INIT  */
  YYSYMBOL_FN_ARG = 24,                    /* FN_ARG  */
  YYSYMBOL_MUL_SUM = 25,                   /* MUL_SUM  */
  YYSYMBOL_DSL_FN = 26,                    /* DSL_FN  */
  YYSYMBOL_DSL_DOT = 27,                   /* DSL_DOT  */
  YYSYMBOL_FFN_OUT = 28,                   /* FFN_OUT  */
  YYSYMBOL_SIZE_FN = 29,                   /* SIZE_FN  */
  YYSYMBOL_RELAXNLN = 30,                  /* RELAXNLN  */
  YYSYMBOL_QUANT = 31,                     /* QUANT  */
  YYSYMBOL_GRAPH_ATTR = 32,                /* GRAPH_ATTR  */
  YYSYMBOL_FEAT_ATTR = 33,                 /* FEAT_ATTR  */
  YYSYMBOL_RELU = 34,                      /* RELU  */
  YYSYMBOL_LABEL_ATTR = 35,                /* LABEL_ATTR  */
  YYSYMBOL_DEGREE_ATTR = 36,               /* DEGREE_ATTR  */
  YYSYMBOL_NODE_ATTR = 37,                 /* NODE_ATTR  */
  YYSYMBOL_LEAKY_RELU = 38,                /* LEAKY_RELU  */
  YYSYMBOL_RABBIT_REORDER_OP = 39,         /* RABBIT_REORDER_OP  */
  YYSYMBOL_SAMPLE_RANDOM_OP = 40,          /* SAMPLE_RANDOM_OP  */
  YYSYMBOL_POW = 41,                       /* POW  */
  YYSYMBOL_SCALAR_INIT = 42,               /* SCALAR_INIT  */
  YYSYMBOL_COLTILE = 43,                   /* COLTILE  */
  YYSYMBOL_AGGR = 44,                      /* AGGR  */
  YYSYMBOL_FEAT_SIZE_ASSIGN = 45,          /* FEAT_SIZE_ASSIGN  */
  YYSYMBOL_LABEL_SIZE_ASSIGN = 46,         /* LABEL_SIZE_ASSIGN  */
  YYSYMBOL_COARSEN = 47,                   /* COARSEN  */
  YYSYMBOL_SRC_ATTR = 48,                  /* SRC_ATTR  */
  YYSYMBOL_DST_ATTR = 49,                  /* DST_ATTR  */
  YYSYMBOL_INTEGER = 50,                   /* INTEGER  */
  YYSYMBOL_FLOAT = 51,                     /* FLOAT  */
  YYSYMBOL_SOFTMAX = 52,                   /* SOFTMAX  */
  YYSYMBOL_INIT_WEIGHT = 53,               /* INIT_WEIGHT  */
  YYSYMBOL_LBRACE = 54,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 55,                    /* RBRACE  */
  YYSYMBOL_LSQBRA = 56,                    /* LSQBRA  */
  YYSYMBOL_RSQBRA = 57,                    /* RSQBRA  */
  YYSYMBOL_DOT = 58,                       /* DOT  */
  YYSYMBOL_COMMA = 59,                     /* COMMA  */
  YYSYMBOL_IF = 60,                        /* IF  */
  YYSYMBOL_ELSE = 61,                      /* ELSE  */
  YYSYMBOL_DO = 62,                        /* DO  */
  YYSYMBOL_WHILE = 63,                     /* WHILE  */
  YYSYMBOL_TR = 64,                        /* TR  */
  YYSYMBOL_FA = 65,                        /* FA  */
  YYSYMBOL_NOT = 66,                       /* NOT  */
  YYSYMBOL_AND = 67,                       /* AND  */
  YYSYMBOL_OR = 68,                        /* OR  */
  YYSYMBOL_NOTEQ = 69,                     /* NOTEQ  */
  YYSYMBOL_EQ = 70,                        /* EQ  */
  YYSYMBOL_GREATER = 71,                   /* GREATER  */
  YYSYMBOL_LESS = 72,                      /* LESS  */
  YYSYMBOL_GREATEREQ = 73,                 /* GREATEREQ  */
  YYSYMBOL_LESSEQ = 74,                    /* LESSEQ  */
  YYSYMBOL_PLUS = 75,                      /* PLUS  */
  YYSYMBOL_MINUS = 76,                     /* MINUS  */
  YYSYMBOL_MULTIPLY = 77,                  /* MULTIPLY  */
  YYSYMBOL_DIVIDE = 78,                    /* DIVIDE  */
  YYSYMBOL_FFN = 79,                       /* FFN  */
  YYSYMBOL_DATASET = 80,                   /* DATASET  */
  YYSYMBOL_NONLN = 81,                     /* NONLN  */
  YYSYMBOL_SENSEI_OP = 82,                 /* SENSEI_OP  */
  YYSYMBOL_INT = 83,                       /* INT  */
  YYSYMBOL_NEW = 84,                       /* NEW  */
  YYSYMBOL_NULL_KEY = 85,                  /* NULL_KEY  */
  YYSYMBOL_YYACCEPT = 86,                  /* $accept  */
  YYSYMBOL_program = 87,                   /* program  */
  YYSYMBOL_load_dataset = 88,              /* load_dataset  */
  YYSYMBOL_algorithm = 89,                 /* algorithm  */
  YYSYMBOL_statements = 90,                /* statements  */
  YYSYMBOL_statement = 91,                 /* statement  */
  YYSYMBOL_layers = 92,                    /* layers  */
  YYSYMBOL_layer_def = 93,                 /* layer_def  */
  YYSYMBOL_model = 94,                     /* model  */
  YYSYMBOL_model_def = 95,                 /* model_def  */
  YYSYMBOL_layer_inits = 96,               /* layer_inits  */
  YYSYMBOL_layer_init = 97,                /* layer_init  */
  YYSYMBOL_model_init = 98,                /* model_init  */
  YYSYMBOL_model_uses = 99,                /* model_uses  */
  YYSYMBOL_model_use = 100,                /* model_use  */
  YYSYMBOL_gnn_op = 101,                   /* gnn_op  */
  YYSYMBOL_function = 102,                 /* function  */
  YYSYMBOL_update_op = 103,                /* update_op  */
  YYSYMBOL_schedules = 104,                /* schedules  */
  YYSYMBOL_schedule = 105,                 /* schedule  */
  YYSYMBOL_data_transform = 106,           /* data_transform  */
  YYSYMBOL_function_transform = 107,       /* function_transform  */
  YYSYMBOL_data_var = 108,                 /* data_var  */
  YYSYMBOL_function_init = 109,            /* function_init  */
  YYSYMBOL_semiring_op = 110,              /* semiring_op  */
  YYSYMBOL_op = 111,                       /* op  */
  YYSYMBOL_train_args = 112,               /* train_args  */
  YYSYMBOL_train_arg = 113,                /* train_arg  */
  YYSYMBOL_args = 114,                     /* args  */
  YYSYMBOL_arg = 115,                      /* arg  */
  YYSYMBOL_bool = 116,                     /* bool  */
  YYSYMBOL_string = 117                    /* string  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  5
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   244

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  86
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  81
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  224

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   340


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   349,   349,   372,   403,   404,   408,   413,   414,   420,
     565,   571,   593,   599,   600,   603,   616,   618,   665,   666,
     668,   670,   672,   676,   677,   679,   690,   694,   710,   717,
     723,   728,   734,   739,   744,   750,   754,   758,   763,   764,
     769,   772,   776,   787,   796,   805,   816,   846,   861,   864,
     872,   877,   882,   887,   892,   897,   902,   908,   911,   914,
     914,   914,   914,   916,   917,   920,   925,   927,   929,   932,
     933,   937,   937,   941,   941,   943,   943,   947,   947,   952,
     952,   953
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "IDENTIFIER", "ASSIGN",
  "LOAD", "LPAREN", "RPAREN", "SEMICOLON", "QUOTE", "COMMENT",
  "SET_UNWEIGHTED", "SET_UNDIRECTED", "MODEL_W", "EVAL", "TRAIN", "LAYER",
  "LOSS", "OPTIMIZER", "ITERS", "VAL_STEP", "RMSE_LOSS", "ADAM_T",
  "AGGR_INIT", "FN_ARG", "MUL_SUM", "DSL_FN", "DSL_DOT", "FFN_OUT",
  "SIZE_FN", "RELAXNLN", "QUANT", "GRAPH_ATTR", "FEAT_ATTR", "RELU",
  "LABEL_ATTR", "DEGREE_ATTR", "NODE_ATTR", "LEAKY_RELU",
  "RABBIT_REORDER_OP", "SAMPLE_RANDOM_OP", "POW", "SCALAR_INIT", "COLTILE",
  "AGGR", "FEAT_SIZE_ASSIGN", "LABEL_SIZE_ASSIGN", "COARSEN", "SRC_ATTR",
  "DST_ATTR", "INTEGER", "FLOAT", "SOFTMAX", "INIT_WEIGHT", "LBRACE",
  "RBRACE", "LSQBRA", "RSQBRA", "DOT", "COMMA", "IF", "ELSE", "DO",
  "WHILE", "TR", "FA", "NOT", "AND", "OR", "NOTEQ", "EQ", "GREATER",
  "LESS", "GREATEREQ", "LESSEQ", "PLUS", "MINUS", "MULTIPLY", "DIVIDE",
  "FFN", "DATASET", "NONLN", "SENSEI_OP", "INT", "NEW", "NULL_KEY",
  "$accept", "program", "load_dataset", "algorithm", "statements",
  "statement", "layers", "layer_def", "model", "model_def", "layer_inits",
  "layer_init", "model_init", "model_uses", "model_use", "gnn_op",
  "function", "update_op", "schedules", "schedule", "data_transform",
  "function_transform", "data_var", "function_init", "semiring_op", "op",
  "train_args", "train_arg", "args", "arg", "bool", "string", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-101)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
       5,    29,    52,    41,    56,  -101,    74,    18,    41,    76,
    -101,    81,    96,  -101,    84,    92,    18,  -101,  -101,  -101,
       7,  -101,    97,  -101,  -101,   102,   101,     8,   105,   107,
      62,   109,    -8,    58,   110,    71,    72,  -101,   122,   116,
      35,   125,   123,   134,   132,   122,   114,  -101,   126,   141,
     148,   149,   150,   151,   152,   153,  -101,  -101,  -101,  -101,
    -101,  -101,   122,   122,  -101,   155,   156,   108,   154,  -101,
    -101,  -101,   161,  -101,  -101,  -101,   162,   166,    14,  -101,
     123,   163,   165,    15,    -4,     0,   167,   122,   122,   122,
     120,   122,   168,   122,   118,   118,   169,   170,    95,   172,
     173,  -101,   175,   171,   174,  -101,  -101,  -101,  -101,   122,
    -101,   146,   129,   157,   127,   128,   -30,  -101,   158,    13,
      58,   -16,   177,    19,  -101,    -1,  -101,  -101,   182,   184,
     187,   188,  -101,  -101,     3,  -101,   137,   190,    25,   191,
    -101,   139,  -101,  -101,  -101,   176,  -101,   122,   159,  -101,
    -101,   178,    44,    44,   164,   179,   145,     9,   186,  -101,
    -101,   195,    20,  -101,  -101,   196,    33,   197,   122,  -101,
    -101,   198,   200,   201,   202,  -101,   203,   206,    27,   205,
     211,  -101,  -101,  -101,  -101,  -101,    34,   208,   209,   210,
     212,    21,  -101,  -101,   213,   215,   218,  -101,  -101,    22,
    -101,  -101,  -101,  -101,  -101,   219,  -101,  -101,    10,  -101,
     180,   181,   221,   217,   183,   185,   220,  -101,  -101,  -101,
    -101,    12,   224,  -101
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     4,     0,     1,     0,     0,     4,     0,
      13,     0,     0,    48,     0,     0,     2,    38,    40,    41,
       0,     5,     0,    14,     6,     0,     0,    48,     0,     0,
       0,     0,    26,     0,     0,     0,     0,    39,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    69,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     9,    59,    60,
      61,    62,     0,     0,    12,     0,     0,     0,     0,    51,
      50,    52,     0,    49,    55,    56,     0,     0,     0,    16,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    27,    25,     0,     0,     0,     0,
       0,    69,     0,     0,     0,    22,    81,     3,    29,     0,
      10,     0,     0,     0,    72,    74,    76,    70,     0,     0,
       0,     0,     0,     0,    37,     0,    44,    45,     0,     0,
       0,     0,    53,    54,     0,    69,     0,     0,     0,     0,
       7,    77,    71,    73,    75,     0,    32,     0,     0,    35,
      36,     0,     0,     0,     0,     0,     0,     0,     0,    63,
      28,     0,     0,    78,    58,     0,     0,     0,     0,    79,
      80,     0,     0,     0,     0,    18,     0,     0,     0,     0,
       0,    15,     8,    57,    33,    34,     0,     0,     0,     0,
       0,     0,    21,    69,     0,     0,     0,    64,    11,     0,
      31,    43,    42,    46,    47,     0,    17,    19,     0,    24,
       0,     0,     0,     0,    65,    67,     0,    23,    66,    68,
      69,     0,     0,    20
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -101,  -101,  -101,   225,  -101,    65,  -101,   226,  -101,  -101,
    -101,  -101,  -101,  -101,   160,  -101,  -101,  -101,  -101,   222,
    -101,  -101,    -7,  -101,  -101,   -31,  -101,  -101,  -100,  -101,
      75,  -101
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     2,     3,     7,   162,     8,     9,    10,    24,    25,
     191,   207,    42,    79,    80,    31,    32,    56,    16,    17,
      18,    19,   116,    34,   165,    62,   178,   197,    85,   117,
     171,    44
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      20,   134,    63,    13,   110,    33,    13,   112,     1,    20,
     156,    38,    13,    13,    45,    13,   176,   213,   103,   222,
     146,    13,   108,   180,   205,    27,   150,   113,    39,   144,
     113,    67,   160,     4,   194,   157,   113,   113,    83,   113,
     184,   200,    39,   148,     6,    29,   195,   196,    76,    30,
     114,    28,     5,   114,   111,    94,    95,    39,   151,   114,
     114,    11,   114,    14,    15,    39,    46,    58,    59,    60,
      61,    39,   104,    39,   109,   181,   206,    39,    12,    22,
     119,   120,   121,    39,   123,   115,   125,    26,   115,   147,
      35,    39,    39,   208,   115,   115,    49,   115,    36,    27,
      50,    40,   138,    51,    52,    41,   128,   129,   169,   170,
      43,    47,    28,    48,    53,    54,    39,    57,    64,    29,
     221,    65,    66,    30,    68,    13,    78,    69,    70,    77,
      71,    72,    73,    58,    59,    60,    61,    81,   130,    82,
     166,    55,   131,    74,    75,    68,    84,    87,    69,    70,
      86,    71,    72,    73,    88,    89,    90,    91,    92,    93,
      99,   186,    96,    97,    74,    75,    98,   100,   101,   102,
     122,   118,   106,   107,   136,   124,    39,   126,   127,   132,
     133,   135,   139,   140,   149,   145,   142,   143,   152,   137,
     153,   141,    33,   154,   155,   158,   159,   161,   163,   175,
     177,   164,   179,   183,   185,   187,   168,   188,   189,   190,
     167,   192,   193,   198,   173,   199,   201,   202,   203,   210,
     204,   209,   211,   212,   216,   217,   220,   182,   172,   174,
     214,   215,   223,    21,     0,    23,     0,     0,    37,     0,
     105,     0,   218,     0,   219
};

static const yytype_int16 yycheck[] =
{
       7,   101,    33,     3,     8,    12,     3,     7,     3,    16,
       7,     4,     3,     3,     6,     3,     7,     7,     4,     7,
       7,     3,     7,     3,     3,     3,     7,    27,    58,    59,
      27,    38,     7,     4,     7,   135,    27,    27,    45,    27,
       7,     7,    58,    59,     3,    23,    19,    20,    13,    27,
      50,    16,     0,    50,    58,    62,    63,    58,    59,    50,
      50,     5,    50,    45,    46,    58,    58,    75,    76,    77,
      78,    58,    58,    58,    59,    55,    55,    58,     4,     3,
      87,    88,    89,    58,    91,    85,    93,     6,    85,   120,
       6,    58,    58,   193,    85,    85,    34,    85,     6,     3,
      38,     4,   109,    41,    42,     3,    11,    12,    64,    65,
       9,     6,    16,     6,    52,    53,    58,     8,     8,    23,
     220,    50,    50,    27,    29,     3,     3,    32,    33,     4,
      35,    36,    37,    75,    76,    77,    78,     3,    43,     7,
     147,    79,    47,    48,    49,    29,    32,     6,    32,    33,
      24,    35,    36,    37,     6,     6,     6,     6,     6,     6,
       6,   168,     7,     7,    48,    49,    58,     6,     6,     3,
      50,     4,     9,     8,     3,     7,    58,     8,     8,     7,
       7,     6,    36,    54,     7,    27,    59,    59,     6,    15,
       6,    34,   199,     6,     6,    58,     6,     6,    59,    54,
      14,    25,     7,     7,     7,     7,    28,     7,     7,     7,
      51,     8,     6,     8,    50,     4,     8,     8,     8,     4,
       8,     8,     4,     4,     3,     8,     6,   162,   153,    50,
      50,    50,     8,     8,    -1,     9,    -1,    -1,    16,    -1,
      80,    -1,    59,    -1,    59
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     3,    87,    88,     4,     0,     3,    89,    91,    92,
      93,     5,     4,     3,    45,    46,   104,   105,   106,   107,
     108,    89,     3,    93,    94,    95,     6,     3,    16,    23,
      27,   101,   102,   108,   109,     6,     6,   105,     4,    58,
       4,     3,    98,     9,   117,     6,    58,     6,     6,    34,
      38,    41,    42,    52,    53,    79,   103,     8,    75,    76,
      77,    78,   111,   111,     8,    50,    50,   108,    29,    32,
      33,    35,    36,    37,    48,    49,    13,     4,     3,    99,
     100,     3,     7,   108,    32,   114,    24,     6,     6,     6,
       6,     6,     6,     6,   108,   108,     7,     7,    58,     6,
       6,     6,     3,     4,    58,   100,     9,     8,     7,    59,
       8,    58,     7,    27,    50,    85,   108,   115,     4,   108,
     108,   108,    50,   108,     7,   108,     8,     8,    11,    12,
      43,    47,     7,     7,   114,     6,     3,    15,   108,    36,
      54,    34,    59,    59,    59,    27,     7,   111,    59,     7,
       7,    59,     6,     6,     6,     6,     7,   114,    58,     6,
       7,     6,    90,    59,    25,   110,   108,    51,    28,    64,
      65,   116,   116,    50,    50,    54,     7,    14,   112,     7,
       3,    55,    91,     7,     7,     7,   108,     7,     7,     7,
       7,    96,     8,     6,     7,    19,    20,   113,     8,     4,
       7,     8,     8,     8,     8,     3,    55,    97,   114,     8,
       4,     4,     4,     7,    50,    50,     3,     8,    59,    59,
       6,   114,     7,     8
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    86,    87,    88,    89,    89,    89,    90,    90,    91,
      91,    91,    91,    92,    92,    93,    94,    95,    96,    96,
      97,    98,    99,   100,   100,   101,   101,   101,   102,   102,
     102,   103,   103,   103,   103,   103,   103,   103,   104,   104,
     105,   105,   106,   106,   106,   106,   106,   107,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   109,   110,   111,
     111,   111,   111,   112,   112,   113,   113,   113,   113,   114,
     114,   115,   115,   115,   115,   115,   115,   115,   115,   116,
     116,   117
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     3,     7,     0,     2,     2,     0,     2,     4,
       6,    10,     4,     1,     2,     9,     3,     9,     0,     2,
       7,     7,     2,     9,     7,     3,     1,     3,     6,     4,
       2,     7,     4,     6,     6,     4,     4,     3,     1,     2,
       1,     1,     9,     9,     5,     5,     9,     9,     1,     3,
       3,     3,     3,     5,     5,     3,     3,     7,     1,     1,
       1,     1,     1,     0,     2,     3,     4,     3,     4,     0,
       2,     2,     1,     2,     1,     2,     1,     2,     3,     1,
       1,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* program: load_dataset algorithm schedules  */
#line 350 "frontend.y"
    {
        program.push_back((yyvsp[-2].forwardNode));
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
#line 1657 "build/frontend.tab.c"
    break;

  case 3: /* load_dataset: IDENTIFIER ASSIGN LOAD LPAREN string RPAREN SEMICOLON  */
#line 373 "frontend.y"
    {
        if (debug == 1) cout << "load graph+feats\n";
        (yyval.forwardNode) = new ForwardNode(POINTWISE, LOAD_OP);
        (yyval.forwardNode)->addParam((yyvsp[-2].sval)); 
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

        (yyval.forwardNode)->addOutputData(featData);
        (yyval.forwardNode)->addOutputData(graphData);
        
        free((yyvsp[-6].sval));
        free((yyvsp[-2].sval));
    }
#line 1689 "build/frontend.tab.c"
    break;

  case 4: /* algorithm: %empty  */
#line 403 "frontend.y"
            { (yyval.trainingLoopNode) = NULL; }
#line 1695 "build/frontend.tab.c"
    break;

  case 5: /* algorithm: statement algorithm  */
#line 405 "frontend.y"
    {
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
#line 1703 "build/frontend.tab.c"
    break;

  case 6: /* algorithm: layers model  */
#line 409 "frontend.y"
    {              // so for now just one layer+model then schedules
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
#line 1711 "build/frontend.tab.c"
    break;

  case 7: /* statements: %empty  */
#line 413 "frontend.y"
             { (yyval.trainingLoopNode) = NULL; }
#line 1717 "build/frontend.tab.c"
    break;

  case 8: /* statements: statements statement  */
#line 415 "frontend.y"
    {
        iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
    }
#line 1725 "build/frontend.tab.c"
    break;

  case 9: /* statement: IDENTIFIER ASSIGN gnn_op SEMICOLON  */
#line 421 "frontend.y"
    {
        // TODO: add some code to verify the aggregate function name matches
        if (string((yyvsp[-1].sval)) == "aggregate"){ // aggregate operation
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
        else if (string((yyvsp[-1].sval)) == "ffn"){ // weight operation
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
        else if (string((yyvsp[-1].sval)) == "pow"){
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
        else if (string((yyvsp[-1].sval)) == "normalization"){
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
        else if (string((yyvsp[-1].sval)) == "relu"){
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
        else if (string((yyvsp[-1].sval)) == "scalarEps-mul"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            mulScalarEPS_CIR(dataNodeMap["Feat1"]);
        }
        else if (string((yyvsp[-1].sval)) == "scalarEps-add"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addScalarEPS_CIR(defaultInput, dataNodeMap["Output-Aggregate"]);
        }
        else if (string((yyvsp[-1].sval)) == "init-l-weight"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addAttentionWeight_L(dataNodeMap["Res1"]);
        }
        else if (string((yyvsp[-1].sval)) == "init-r-weight"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addAttentionWeight_R(dataNodeMap["Res1"]);
        }
        else if (string((yyvsp[-1].sval)) == "leaky-relu"){
            DataNode* defaultInput; // TODO: change edge aggregation to be its own thing, not inside leakyRelu
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addEdgeAggregation_CIR(dataNodeMap["attn-l"], dataNodeMap["attn-r"], dataNodeMap["Graph"]);
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addLeakyReLU(defaultInput, dataNodeMap["Graph"]);
        }
        else if (string((yyvsp[-1].sval)) == "softmax"){
            DataNode* defaultInput;
            if (mainTrainingLoop->getLoopNodeNum() == 0){
                defaultInput = NULL;
            }
            else{
                int lastCIR_idx = mainTrainingLoop->getLoopNodeNum()-1;
                defaultInput = mainTrainingLoop->getNode(lastCIR_idx)->getOutput(0);
            }
            addSoftmax_CIR(defaultInput, dataNodeMap["Graph"]);
        }
        free((yyvsp[-3].sval));
        free((yyvsp[-1].sval));
    }
#line 1874 "build/frontend.tab.c"
    break;

  case 10: /* statement: IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON  */
#line 566 "frontend.y"
    {
        (yyval.forwardNode) = NULL;
        free((yyvsp[-5].sval));
        free((yyvsp[-3].sval));
    }
#line 1884 "build/frontend.tab.c"
    break;

  case 11: /* statement: IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR DOT DEGREE_ATTR LPAREN RPAREN SEMICOLON  */
#line 572 "frontend.y"
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
        free((yyvsp[-9].sval));
        free((yyvsp[-7].sval));
    }
#line 1910 "build/frontend.tab.c"
    break;

  case 12: /* statement: IDENTIFIER ASSIGN function_init SEMICOLON  */
#line 594 "frontend.y"
    {
        (yyval.forwardNode) = NULL;
        free((yyvsp[-3].sval));
    }
#line 1919 "build/frontend.tab.c"
    break;

  case 13: /* layers: layer_def  */
#line 599 "frontend.y"
                   { (yyval.trainingLoopNode) = (yyvsp[0].trainingLoopNode); }
#line 1925 "build/frontend.tab.c"
    break;

  case 14: /* layers: layers layer_def  */
#line 601 "frontend.y"
    {}
#line 1931 "build/frontend.tab.c"
    break;

  case 15: /* layer_def: IDENTIFIER ASSIGN LAYER LPAREN args RPAREN LBRACE statements RBRACE  */
#line 604 "frontend.y"
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
#line 1946 "build/frontend.tab.c"
    break;

  case 16: /* model: model_def model_init model_uses  */
#line 616 "frontend.y"
                                        {}
#line 1952 "build/frontend.tab.c"
    break;

  case 17: /* model_def: IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE  */
#line 618 "frontend.y"
                                                                                   {
    int n = mainTrainingLoop->getLoopNodeNum();
    for (int i = 1; i < numLayers; i++){
        int n_i = mainTrainingLoop->getLoopNodeNum();
        DataNode* lastData = mainTrainingLoop->getNode(n_i-1)->getOutput(0);
        for (int j = 0; j < n; j++){
            ForwardNode* curNode = mainTrainingLoop->getNode(j);
            string type = curNode->getOutput(0)->getName();
            if (type == "ones" || type == "degrees" || type == "norm"){
                lastData = curNode->getOutput(0);
                continue;
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
            else if (type == "res-mulScalarEPS"){
                mulScalarEPS_CIR(dataNodeMap["reluData"]); // TODO: change this so it works for more than 2 layers
            }
            else if (type == "res-addScalarEPS"){
                addScalarEPS_CIR(lastData, dataNodeMap["Output-Aggregate"]); // TODO: change to return newest aggregate node
            }
            else if (type == "attenR"){
                addAttentionWeight_R(dataNodeMap["Res1"]);
            }
            else if (type == "attenL"){
                addAttentionWeight_L(dataNodeMap["Res1"]);
            }
            else if (type == "res-softmax"){
                addSoftmax_CIR(lastData, dataNodeMap["Graph"]);
            }
            else if (type == "res-leakyReLU"){
                addLeakyReLU(lastData, dataNodeMap["Graph"]);
            }
            lastData = curNode->getOutput(0);
        }
    }
}
#line 2003 "build/frontend.tab.c"
    break;

  case 18: /* layer_inits: %empty  */
#line 665 "frontend.y"
              { (yyval.irNode) = NULL; }
#line 2009 "build/frontend.tab.c"
    break;

  case 19: /* layer_inits: layer_inits layer_init  */
#line 666 "frontend.y"
                             {}
#line 2015 "build/frontend.tab.c"
    break;

  case 20: /* layer_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 668 "frontend.y"
                                                                       { numLayers++; free((yyvsp[-6].sval)); free((yyvsp[-4].sval)); }
#line 2021 "build/frontend.tab.c"
    break;

  case 21: /* model_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 670 "frontend.y"
                                                                       { free((yyvsp[-6].sval)); free((yyvsp[-4].sval)); }
#line 2027 "build/frontend.tab.c"
    break;

  case 22: /* model_uses: model_use model_use  */
#line 672 "frontend.y"
                                 { (yyval.irNode) = NULL; }
#line 2033 "build/frontend.tab.c"
    break;

  case 23: /* model_use: IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON  */
#line 676 "frontend.y"
                                                                               { free((yyvsp[-8].sval)); free((yyvsp[-6].sval)); }
#line 2039 "build/frontend.tab.c"
    break;

  case 24: /* model_use: IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON  */
#line 677 "frontend.y"
                                                              { free((yyvsp[-6].sval)); }
#line 2045 "build/frontend.tab.c"
    break;

  case 25: /* gnn_op: data_var op data_var  */
#line 680 "frontend.y"
    {
         if ((yyvsp[-1].ival) == 3){
            (yyval.sval) = strdup("normalization"); 
         }
         else if ((yyvsp[-1].ival) == 1){
            (yyval.sval) = strdup("scalarEps-add");
         }
         free((yyvsp[-2].sval)); 
         free((yyvsp[0].sval)); 
    }
#line 2060 "build/frontend.tab.c"
    break;

  case 26: /* gnn_op: function  */
#line 691 "frontend.y"
    {
        (yyval.sval) = (yyvsp[0].sval);
    }
#line 2068 "build/frontend.tab.c"
    break;

  case 27: /* gnn_op: function op data_var  */
#line 695 "frontend.y"
    {
        if (string((yyvsp[-2].sval)) == "scalarEps-mul");{
            (yyval.sval) = (yyvsp[-2].sval);
        }
        if (string((yyvsp[-2].sval)) == "init-weight"){
            if (string((yyvsp[0].sval)) == "src_nodes"){
                (yyval.sval) = strdup("init-l-weight");
            }
            else{
                (yyval.sval) = strdup("init-r-weight");
            }
        }
        free((yyvsp[0].sval));
    }
#line 2087 "build/frontend.tab.c"
    break;

  case 28: /* function: IDENTIFIER LPAREN data_var COMMA data_var RPAREN  */
#line 711 "frontend.y"
    {
        (yyval.sval) = strdup("aggregate");
        free((yyvsp[-5].sval));
        free((yyvsp[-3].sval));
        free((yyvsp[-1].sval));
    }
#line 2098 "build/frontend.tab.c"
    break;

  case 29: /* function: IDENTIFIER LPAREN data_var RPAREN  */
#line 718 "frontend.y"
    {
        (yyval.sval) = strdup("relu");
        free((yyvsp[-3].sval));
        free((yyvsp[-1].sval));
    }
#line 2108 "build/frontend.tab.c"
    break;

  case 30: /* function: DSL_DOT update_op  */
#line 724 "frontend.y"
    {
        (yyval.sval) = (yyvsp[0].sval);
    }
#line 2116 "build/frontend.tab.c"
    break;

  case 31: /* update_op: FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN  */
#line 729 "frontend.y"
    {
        (yyval.sval) = strdup("ffn");
        free((yyvsp[-4].sval));
        free((yyvsp[-1].sval));
    }
#line 2126 "build/frontend.tab.c"
    break;

  case 32: /* update_op: RELU LPAREN data_var RPAREN  */
#line 735 "frontend.y"
    {
        (yyval.sval) = strdup("relu");
        free((yyvsp[-1].sval));
    }
#line 2135 "build/frontend.tab.c"
    break;

  case 33: /* update_op: LEAKY_RELU LPAREN data_var op data_var RPAREN  */
#line 740 "frontend.y"
    {
        (yyval.sval) = strdup("leaky-relu");
        free((yyvsp[-3].sval));
    }
#line 2144 "build/frontend.tab.c"
    break;

  case 34: /* update_op: POW LPAREN data_var COMMA FLOAT RPAREN  */
#line 745 "frontend.y"
    {
        // TODO: use the float in the calculation, now just hardcoded to -0.5
        (yyval.sval) = strdup("pow");
        free((yyvsp[-3].sval));
    }
#line 2154 "build/frontend.tab.c"
    break;

  case 35: /* update_op: SCALAR_INIT LPAREN INTEGER RPAREN  */
#line 751 "frontend.y"
    {
        (yyval.sval) = strdup("scalarEps-mul");
    }
#line 2162 "build/frontend.tab.c"
    break;

  case 36: /* update_op: SOFTMAX LPAREN data_var RPAREN  */
#line 755 "frontend.y"
    {
        (yyval.sval) = strdup("softmax");
    }
#line 2170 "build/frontend.tab.c"
    break;

  case 37: /* update_op: INIT_WEIGHT LPAREN RPAREN  */
#line 759 "frontend.y"
    {
        (yyval.sval) = strdup("init-weight");
    }
#line 2178 "build/frontend.tab.c"
    break;

  case 38: /* schedules: schedule  */
#line 763 "frontend.y"
                     {}
#line 2184 "build/frontend.tab.c"
    break;

  case 39: /* schedules: schedules schedule  */
#line 765 "frontend.y"
    {

    }
#line 2192 "build/frontend.tab.c"
    break;

  case 40: /* schedule: data_transform  */
#line 770 "frontend.y"
    {
    }
#line 2199 "build/frontend.tab.c"
    break;

  case 41: /* schedule: function_transform  */
#line 773 "frontend.y"
    {
    }
#line 2206 "build/frontend.tab.c"
    break;

  case 42: /* data_transform: data_var ASSIGN data_var DOT SET_UNDIRECTED LPAREN bool RPAREN SEMICOLON  */
#line 777 "frontend.y"
    {
        // if transformed graph already exists, then modify that as well
        // always modify the original graph
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setDirected(!(yyvsp[-2].ival));
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setDirected(!(yyvsp[-2].ival));
    }
#line 2221 "build/frontend.tab.c"
    break;

  case 43: /* data_transform: data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON  */
#line 788 "frontend.y"
    {
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setWeighted(!(yyvsp[-2].ival));
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setWeighted(!(yyvsp[-2].ival));
    }
#line 2234 "build/frontend.tab.c"
    break;

  case 44: /* data_transform: FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 797 "frontend.y"
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat1"]->getData()->next());
        DataInfo* outAggrInfo = dynamic_cast<DataInfo*>(dataNodeMap["Output-Aggregate"]->getData()->next());
        int dimRow = featInfo->getDimRow();
        featInfo->setDims(dimRow, atoi((yyvsp[-2].sval)));
        outAggrInfo->setDims(dimRow, atoi((yyvsp[-2].sval)));
        free((yyvsp[-2].sval));
    }
#line 2247 "build/frontend.tab.c"
    break;

  case 45: /* data_transform: LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 806 "frontend.y"
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat1"]->getData()->next());
        DataInfo* weightInfo = dynamic_cast<DataInfo*>(dataNodeMap["Weight1"]->getData()->next());
        DataInfo* resInfo = dynamic_cast<DataInfo*>(dataNodeMap["Res1"]->getData()->next());
        int featDimRow = featInfo->getDimRow();
        int featDimCol = featInfo->getDimCol();
        weightInfo->setDims(featDimCol, atoi((yyvsp[-2].sval)));
        resInfo->setDims(featDimRow, atoi((yyvsp[-2].sval)));
        free((yyvsp[-2].sval));
    }
#line 2262 "build/frontend.tab.c"
    break;

  case 46: /* data_transform: data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON  */
#line 817 "frontend.y"
    {
        // actually creating new DIR
        DataLevel* originalRootGraphLevel = dataNodeMap["Graph"]->getData();
        // TODO: ask about using DataItem* b/c it is an abstract class, so should either be DataLevel or DataInfo?
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(originalRootGraphLevel->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), originalGraphInfo->getWeighted());
        transformedGraphInfo->addOpt(COL_TILE_DOPT, atoi((yyvsp[-2].sval)));
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
        tileTransformation->addParam((yyvsp[-2].sval));
        TransformEdge* graphTrgraph = new TransformEdge(dataNodeMap["Graph"], transformedGraph);
        graphTrgraph->addTransformation(tileTransformation);
        transforms.push_back(graphTrgraph);
            
        RelationEdge* inOutAggrRelationTrGraph = new RelationEdge(transformedGraph, ALL_RELATION, dataNodeMap["Output-Aggregate"], ALL_RELATION);
        dependencies.push_back(inOutAggrRelationTrGraph);
        free((yyvsp[-6].sval));
    }
#line 2295 "build/frontend.tab.c"
    break;

  case 47: /* function_transform: data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON  */
#line 847 "frontend.y"
    {
        if (computeNodeMap.find("aggregate") != computeNodeMap.end()){
            for (ForwardNode* a : computeNodeMap["aggregate"]){
                a->addOpt(COARSE_COPT, atoi((yyvsp[-2].sval)));
            }
        }
        else{
            cout << "error\n";
        }
        free((yyvsp[-2].sval));
    }
#line 2311 "build/frontend.tab.c"
    break;

  case 48: /* data_var: IDENTIFIER  */
#line 862 "frontend.y"
    {
    }
#line 2318 "build/frontend.tab.c"
    break;

  case 49: /* data_var: data_var DOT NODE_ATTR  */
#line 865 "frontend.y"
    {
        if ((yyvsp[-2].sval) == "feats"){
            (yyval.sval) = (yyvsp[-2].sval);
        }
        (yyval.sval) = strdup("node");
        free((yyvsp[-2].sval));
    }
#line 2330 "build/frontend.tab.c"
    break;

  case 50: /* data_var: data_var DOT FEAT_ATTR  */
#line 873 "frontend.y"
    {
        (yyval.sval) = strdup("feats");
        free((yyvsp[-2].sval));
    }
#line 2339 "build/frontend.tab.c"
    break;

  case 51: /* data_var: data_var DOT GRAPH_ATTR  */
#line 878 "frontend.y"
    {
        (yyval.sval) = strdup("graphs");
        free((yyvsp[-2].sval));
    }
#line 2348 "build/frontend.tab.c"
    break;

  case 52: /* data_var: data_var DOT LABEL_ATTR  */
#line 883 "frontend.y"
    {
        (yyval.sval) = strdup("label");
        free((yyvsp[-2].sval));
    }
#line 2357 "build/frontend.tab.c"
    break;

  case 53: /* data_var: data_var DOT SIZE_FN LPAREN RPAREN  */
#line 888 "frontend.y"
    {
        (yyval.sval) = strdup("size");
        free((yyvsp[-4].sval));
    }
#line 2366 "build/frontend.tab.c"
    break;

  case 54: /* data_var: data_var DOT DEGREE_ATTR LPAREN RPAREN  */
#line 893 "frontend.y"
    {
        (yyval.sval) = strdup("degrees");
        free((yyvsp[-4].sval));
    }
#line 2375 "build/frontend.tab.c"
    break;

  case 55: /* data_var: data_var DOT SRC_ATTR  */
#line 898 "frontend.y"
    {
        (yyval.sval) = strdup("src_nodes");
        free((yyvsp[-2].sval));
    }
#line 2384 "build/frontend.tab.c"
    break;

  case 56: /* data_var: data_var DOT DST_ATTR  */
#line 903 "frontend.y"
    {
        (yyval.sval) = strdup("dst_nodes");
        free((yyvsp[-2].sval));
    }
#line 2393 "build/frontend.tab.c"
    break;

  case 57: /* function_init: AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN  */
#line 909 "frontend.y"
    {}
#line 2399 "build/frontend.tab.c"
    break;

  case 58: /* semiring_op: MUL_SUM  */
#line 912 "frontend.y"
    {}
#line 2405 "build/frontend.tab.c"
    break;

  case 59: /* op: PLUS  */
#line 914 "frontend.y"
          { (yyval.ival) = 1; }
#line 2411 "build/frontend.tab.c"
    break;

  case 60: /* op: MINUS  */
#line 914 "frontend.y"
                              { (yyval.ival) = 2; }
#line 2417 "build/frontend.tab.c"
    break;

  case 61: /* op: MULTIPLY  */
#line 914 "frontend.y"
                                                     { (yyval.ival) = 3; }
#line 2423 "build/frontend.tab.c"
    break;

  case 62: /* op: DIVIDE  */
#line 914 "frontend.y"
                                                                          { (yyval.ival) = 4; }
#line 2429 "build/frontend.tab.c"
    break;

  case 63: /* train_args: %empty  */
#line 916 "frontend.y"
             { (yyval.irNode) = NULL; }
#line 2435 "build/frontend.tab.c"
    break;

  case 64: /* train_args: train_args train_arg  */
#line 918 "frontend.y"
    {}
#line 2441 "build/frontend.tab.c"
    break;

  case 65: /* train_arg: ITERS ASSIGN INTEGER  */
#line 921 "frontend.y"
    {
        trainArgs["iters"] = atoi((yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 2450 "build/frontend.tab.c"
    break;

  case 66: /* train_arg: ITERS ASSIGN INTEGER COMMA  */
#line 926 "frontend.y"
    { trainArgs["iters"] = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 2456 "build/frontend.tab.c"
    break;

  case 67: /* train_arg: VAL_STEP ASSIGN INTEGER  */
#line 928 "frontend.y"
    { trainArgs["val_step"] = atoi((yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 2462 "build/frontend.tab.c"
    break;

  case 68: /* train_arg: VAL_STEP ASSIGN INTEGER COMMA  */
#line 930 "frontend.y"
    { trainArgs["val_step"] = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 2468 "build/frontend.tab.c"
    break;

  case 69: /* args: %empty  */
#line 932 "frontend.y"
       { (yyval.irNode) = NULL; }
#line 2474 "build/frontend.tab.c"
    break;

  case 70: /* args: args arg  */
#line 934 "frontend.y"
    {
    }
#line 2481 "build/frontend.tab.c"
    break;

  case 71: /* arg: INTEGER COMMA  */
#line 937 "frontend.y"
                    { free((yyvsp[-1].sval)); }
#line 2487 "build/frontend.tab.c"
    break;

  case 72: /* arg: INTEGER  */
#line 938 "frontend.y"
    {
        free((yyvsp[0].sval));
    }
#line 2495 "build/frontend.tab.c"
    break;

  case 74: /* arg: NULL_KEY  */
#line 942 "frontend.y"
    {}
#line 2501 "build/frontend.tab.c"
    break;

  case 75: /* arg: data_var COMMA  */
#line 943 "frontend.y"
                     { free((yyvsp[-1].sval)); }
#line 2507 "build/frontend.tab.c"
    break;

  case 76: /* arg: data_var  */
#line 944 "frontend.y"
    {
        free((yyvsp[0].sval));
    }
#line 2515 "build/frontend.tab.c"
    break;

  case 78: /* arg: DSL_DOT RELU COMMA  */
#line 948 "frontend.y"
    {

    }
#line 2523 "build/frontend.tab.c"
    break;

  case 79: /* bool: TR  */
#line 952 "frontend.y"
          { (yyval.ival) = 1; }
#line 2529 "build/frontend.tab.c"
    break;

  case 80: /* bool: FA  */
#line 952 "frontend.y"
                           { (yyval.ival) = 2; }
#line 2535 "build/frontend.tab.c"
    break;

  case 81: /* string: QUOTE IDENTIFIER QUOTE  */
#line 954 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 2545 "build/frontend.tab.c"
    break;


#line 2549 "build/frontend.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 961 "frontend.y"


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
    debug = 1;
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
