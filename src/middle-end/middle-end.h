//
// Created by damitha on 12/28/24.
//

#ifndef MIDDLE_END_H
#define MIDDLE_END_H

#include <string>
#include <string>
#include <vector>
#include "../ir/compute.h"
#include <fstream>
#include <iostream>
#include <algorithm>

class GALATransformations
{
public:
    // TODO Training aware subgraph creation
    //  Changes to the IR - Create the transformed graphs and change the outputs
    //  Changes to the code generation - Should change based on the input, create the code for the transformed data

    /**
     * Assumptions / Rules
     *  - The dense input would always be written to res (so you can use that directly for codegen)
     *   - BUT! any changes would also change the input / outputs for a particular function and the dependencies
     */

    static int getEdgeIndex(std::vector<RelationEdge*> edges, DataNode *n1, DataNode *n2)
    {
        // This is a very bad implementation for now. Use a KV type of lookup to make it efficient
        for (int ix = 0; ix < edges.size(); ix++)
        {
            auto edge = edges[ix];
            if (edge->getNode1() == n1 && edge->getNode2() == n2)
            {
                return ix;
            }
        }
        return -1;
    }

    // DataNode *createGraphCopyWithIndex(ComputeNode* cNode, int index)
    // {
    //     auto graphInput =  cNode->getInput(1);
    //     auto graphInfo = graphInput->getDataInfo();
    //     auto newInfo = DataInfo(graphInfo->getFormat(), graphInfo->getDirected(), graphInfo->getWeighted());
    //     // Add all the existing optimizations
    //     for (auto opt: graphInfo->getOpts())
    //     {
    //         newInfo.addOpt(opt.first, opt.second);
    //     }
    // }

    static void trainingSubGraph(std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        int countAggregations = 0;
        DataNode* initialGraphNode;
        DataNode* transformedGraphNode;

        // Iterate through program to locate the training loop
        for (int i = 0; i < program.size(); i++)
        {
            std::string outName = "";
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                // TODO Create the initial input graph.
                //  i.e. add computations to the IR to create these results

                for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                {
                    // TODO Change this to a stringed set of aggregations not just aggregation
                    //  i.e. result of aggregation 1 is used by aggregation 2
                    //  if the aggregations are separate then they can still be independent
                    auto inNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix));
                    if (inNode->getOp() == AGGREGATE_MUL_SUM_OP)
                    {
                        if (outName == "")
                        {
                            countAggregations++;
                            outName = inNode->getOutput(0)->getName();
                            transformedGraphNode = inNode->getInput(1);
                        } else
                        {
                            if (outName == inNode->getInput(0)->getName())
                            {
                                countAggregations++;
                            }
                        }
                    }
                }
            } else
            {
                auto inNode = dynamic_cast<ComputeNode*>(outNode);
                if (inNode->getOp() == AGGREGATE_MUL_SUM_OP)
                {
                    countAggregations++;
                    // TODO Add index to be used by the subgraph operation and copy all the transformations in it.
                } else if (inNode->getOp() == LOAD_OP)
                {
                    initialGraphNode = inNode->getOutput(1);
                }
            }
        }

        bool graphTransformed = false;
        if (initialGraphNode != transformedGraphNode)
        {
            graphTransformed = true;
        }

        std::vector<DataNode*> finalGraphs;

        // Add new transformation to the original graph (Also need to create sub-objects, these would then process
        // their transformations)
        for (int ic = 1 ; ic < countAggregations + 1; ic++)
        {
            // TODO Find a better way to clone
            auto graphInfo = new DataInfo(CSR_STYPE, true, initialGraphNode->getDataInfo()->getWeighted());
            graphInfo->setDefaultName("adj");
            graphInfo->setIndex(ic);
            graphInfo->setDefaultDirected(initialGraphNode->getDataInfo()->getDirected());
            auto rootGraphLevel = new DataLevel(graphInfo, true);
            auto newGraph = new DataNode("adj"+std::to_string(ic), INT32, INT32, F32, rootGraphLevel);

            auto subGraphTransformation = new TransformData(SUBGRAPH_DOPT);
            subGraphTransformation->addParam(std::to_string(ic));
            if (ic == 1)
            {
                subGraphTransformation->addParam(std::to_string(countAggregations));
            }
            // else
            // {
            //     std::cout << "Subgraph param: " << ic << std::endl;
            // }
            auto graphSubgraph = new TransformEdge(initialGraphNode, newGraph);
            graphSubgraph->addTransformation(subGraphTransformation);
            transforms.push_back(graphSubgraph);

            // Rough solution for now. If this is a transformed graph, then just add new
            if (graphTransformed)
            {
                auto opt = transformedGraphNode->getDataInfo()->getOpts()->at(0);

                auto transformedGraphInfo = new DataInfo(CSR_STYPE, true,
                    transformedGraphNode->getDataInfo()->getWeighted());
                transformedGraphInfo->addOpt(opt.first, opt.second);
                transformedGraphInfo->setIndex(ic);
                transformedGraphInfo->setDefaultName("graph_tile");
                transformedGraphInfo->setDefaultDirected(transformedGraphNode->getDataInfo()->getDirected());
                auto transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
                auto transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
                auto newTransformedGraph = new DataNode("graph_tile"+std::to_string(ic),
                    transformedGraphNode->getIType(),
                    transformedGraphNode->getNType(),
                    transformedGraphNode->getVType(),
                    transformedRootGraphLevel);

                auto subTrGraphCopyTransformation = new TransformData(opt.first);
                // subTrGraphCopyTransformation->addParam(std::to_string(ic));
                subTrGraphCopyTransformation->addParam(opt.second);
                auto graphTrSubgraph = new TransformEdge(newGraph, newTransformedGraph);
                graphTrSubgraph->addTransformation(subTrGraphCopyTransformation);
                transforms.push_back(graphTrSubgraph);
                finalGraphs.push_back(newTransformedGraph);
            }
        }

        countAggregations = 0;
        // This pass actually assigns the subgraphs and the transformed graphs
        for (int i = 0; i < program.size(); i++)
        {
            std::string outName = "";
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                // TODO Create the initial input graph.
                //  i.e. add computations to the IR to create these results

                for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                {
                    // TODO Change this to a stringed set of aggregations not just aggregation
                    //  i.e. result of aggregation 1 is used by aggregation 2
                    //  if the aggregations are separate then they can still be independent
                    auto inNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix));
                    if (inNode->getOp() == AGGREGATE_MUL_SUM_OP)
                    {
                        // TODO make all aggregations increment
                        inNode->setInputDataNode(1, finalGraphs.at(countAggregations));
                        countAggregations++;
                        // if (outName == "")
                        // {
                        //     inNode->setInputDataNode(1, finalGraphs.at(countAggregations));
                        //     countAggregations++;
                        // } else
                        // {
                        //     if (outName == inNode->getInput(0)->getName())
                        //     {
                        //         inNode->setInputDataNode(1, finalGraphs.at(countAggregations));
                        //         countAggregations++;
                        //     }
                        // }
                    }
                }
            } else
            {
                auto inNode = dynamic_cast<ComputeNode*>(outNode);
                if (inNode->getOp() == AGGREGATE_MUL_SUM_OP)
                {
                    inNode->setInputDataNode(1, finalGraphs.at(countAggregations));
                    countAggregations++;
                    // TODO Add index to be used by the subgraph operation and copy all the transformations in it.
                }
            }
        }
    }

    // TODO have a commong way to write re-write rules.
    // TODO Sparsity aware re-write
    static void sparsityAwareRewrites(std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        // Do complexity aware re-ordering to make the learning parts away from the other graph computations
        complexityOperatorReordering(program, dependencies, associations, transforms, false);
        // Iterate through program to locate the training loop
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                // TODO Create the initial input graph.
                //  i.e. add computations to the IR to create these results

                for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                {
                    // TODO Change this to a stringed set of aggregations not just aggregation
                    //  i.e. result of aggregation 1 is used by aggregation 2
                    //  if the aggregations are separate then they can still be independent
                    if (ix + 2 < lNode->getLoopNodeNum())
                    {
                        auto cNodeRb1 = dynamic_cast<ComputeNode*>(lNode->getNode(ix));
                        auto cNodeAggr = dynamic_cast<ComputeNode*>(lNode->getNode(ix + 1));
                        auto cNodeRb2 = dynamic_cast<ComputeNode*>(lNode->getNode(ix + 2));

                        if(cNodeAggr->getInput(1)->getDataInfo()->getSparse() &&
                            cNodeRb1->getOp() == ROW_BROADCAST_OP &&
                            cNodeRb2->getOp() == ROW_BROADCAST_OP &&
                            cNodeAggr->getOp() == AGGREGATE_MUL_SUM_OP)
                        {
                            // TODO remove dependencies
                            lNode->eraseFirstNLoopNodes(3, ix);

                            auto inputGraph = cNodeAggr->getInput(1);
                            // Edge aggregation
                            auto aggregateEdge = ForwardNode(AGGREGATE_EDGE, AGGREGATE_EDGE_MUL_OP);
                            auto aggrEdgeInfo = DataInfo(CSR_STYPE, inputGraph->getDataInfo()->getDirected(), true);
                            auto rootAggrEdgeLevel = DataLevel(&aggrEdgeInfo, true);
                            auto aggrEdgeData = DataNode("val", INT32, INT32, F32, &rootAggrEdgeLevel);
                            aggregateEdge.addInputData(cNodeRb1->getInput(0));
                            aggregateEdge.addInputData(cNodeRb2->getInput(0));
                            aggregateEdge.addInputData(inputGraph);
                            aggregateEdge.addOutputData(&aggrEdgeData);
                            lNode->insertToLoopNodes(ix, &aggregateEdge);
                            //* Dependencies
                            // Dependency relation between the features and the aggregated output
                            auto inOutEdgeAggrLRelationFeat = RelationEdge(cNodeRb1->getInput(0), ALL_RELATION, &aggrEdgeData, ROWS_RELATION);
                            auto inOutEdgeAggrRRelationFeat = RelationEdge(cNodeRb2->getInput(0), ALL_RELATION, &aggrEdgeData, COLS_RELATION);
                            auto inOutEdgeAggrRelationGraph = RelationEdge(inputGraph, ALL_RELATION, &aggrEdgeData, ALL_RELATION);
                            dependencies.push_back(&inOutEdgeAggrLRelationFeat);
                            dependencies.push_back(&inOutEdgeAggrRRelationFeat);
                            dependencies.push_back(&inOutEdgeAggrRelationGraph);
                            auto graphEdgeAggrLAssociation = RelationEdge(inputGraph, ROWS_RELATION, cNodeRb1->getInput(0), ALL_RELATION);
                            auto graphEdgeAggrRAssociation = RelationEdge(inputGraph, COLS_RELATION, cNodeRb2->getInput(0), ALL_RELATION);
                            associations.push_back(&graphEdgeAggrLAssociation);
                            associations.push_back(&graphEdgeAggrRAssociation);

                            // Add aggregate operation
                            auto aggregate = ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
                            auto outputInfo = DataInfo(RM_DTYPE);
                            outputInfo.setDims(-1, 32); // -1=N=232965, the number of nodes in the graph
                            auto rootOutputLevel = DataLevel(&outputInfo, true);
                            auto outputData = DataNode("res", INT32, INT32, F32, &rootOutputLevel);
                            aggregate.addInputData(cNodeRb1->getInput(1));
                            aggregate.addInputData(&aggrEdgeData);
                            aggregate.addOutputData(&outputData);
                            aggregate.addOpt(COARSE_COPT, 2);
                            lNode->insertToLoopNodes( ix + 1, &aggregate);
                            //* Dependencies
                            // Dependency relation between the features and the aggregated output
                            auto inOutAggrRelationFeat = RelationEdge(cNodeRb1->getInput(1), ALL_RELATION, &outputData, ALL_RELATION);
                            // Dependency relation between the graph and the aggregated output
                            auto inOutAggrRelationGraph = RelationEdge(&aggrEdgeData, ALL_RELATION, &outputData, ROWS_RELATION);
                            dependencies.push_back(&inOutAggrRelationFeat);
                            dependencies.push_back(&inOutAggrRelationGraph);
                        }
                    }

                    auto cNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix));
                    if(cNode->getOp() == FFN_OP)
                    {
                        // Get output and then get places the output is used.
                        // If one of those places is a aggregate operation, if going from smaller to larger,
                        // and is sparse, then do recompute
                        // TODO add the computation that adds the dependency to the object?
                        int outputUses = 0;
                        auto output = cNode->getOutput(0);
                        auto inputDataInfo = cNode->getInput(0)->getDataInfo();
                        auto inputCols = inputDataInfo->getDimCol();
                        auto outputCols = output->getDataInfo()->getDimCol();
                        for (int iy = ix; iy < lNode->getLoopNodeNum(); iy++)
                        {
                            auto oNode = dynamic_cast<ComputeNode*>(lNode->getNode(iy));
                            if (outputUses > 0 &&
                                oNode->getOp() == AGGREGATE_MUL_SUM_OP &&
                                inputCols < outputCols &&
                                !oNode->getInput(1)->getDataInfo()->getSparse())
                            {
                                oNode->setInputDataNode(0, cNode->getInput(0));
                                oNode->getOutput(0)->getDataInfo()->setDims(inputDataInfo->getDimRow(), inputCols);

                                // Add weight operation
                                auto ffn = ForwardNode(UPDATE_NODE, FFN_OP);
                                // Res DIR
                                auto resInfo = DataInfo(RM_DTYPE);
                                resInfo.setDims(-1, outputCols); // -1=N=232965, the number of nodes in the graph, -3=output classes
                                auto rootResLevel = DataLevel(&resInfo, true);
                                auto resData = DataNode("res", INT32, INT32, F32, &rootResLevel);
                                ffn.addInputData(oNode->getOutput(0));
                                ffn.addInputData(cNode->getInput(1));
                                ffn.addOutputData(&resData);
                                lNode->insertToLoopNodes(iy + 1, &ffn);
                                //* Dependencies
                                auto inOutWeightDepRelationFeat = RelationEdge(oNode->getOutput(0), ALL_RELATION, &resData, ALL_RELATION);
                                auto inOutWeightDepRelationWeight = RelationEdge(cNode->getInput(1), COLS_RELATION, &resData, ROWS_RELATION);
                                dependencies.push_back(&inOutWeightDepRelationFeat);
                                dependencies.push_back(&inOutWeightDepRelationWeight);
                                auto inOutWeightAssociation = RelationEdge(oNode->getOutput(0), ROWS_RELATION, cNode->getInput(1), COLS_RELATION);
                                associations.push_back(&inOutWeightAssociation);
                            }
                            if (oNode->getInput(0) == output)
                            {
                                outputUses++;
                            }
                        }
                    }
                }
            } else {
                // TODO Ignore this part for now. Assume all computations come from the main function
            }
        }

    }

    // TODO Training invariant code motion
    static void trainingInvariantCodeMotion (std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        // Do complexity aware re-ordering to make the learning parts as far off in the computation as possible.
        complexityOperatorReordering(program, dependencies, associations, transforms, true);

        bool foundLeaning = false;
        int nodesMoved = 0;
        std::string inputName = "";
        // Iterate through the list of operations in the training loop till you hit a learned operation
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                int ix = 0;
                while (ix < lNode->getLoopNodeNum())
                {
                    auto cNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix));
                    if (cNode->getOp() == FFN_OP || cNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
                    {
                        cNode->getInput(0)->setName(inputName);
                        foundLeaning = true;
                        break;
                    } else
                    {
                        // Add and remove
                        program.insert(program.begin() + i + nodesMoved, cNode);
                        if (inputName == "")
                        {
                            if (cNode->getOp() == AGGREGATE_MUL_SUM_OP)
                            {
                                inputName = cNode->getInput(0)->getName();
                                std::string outputName = cNode->getOutput(0)->getName();
                                // Change this later on
                                if (outputName == "res_n")
                                {
                                    int ixx = ix + 1;
                                    while (ixx < lNode->getLoopNodeNum())
                                    {
                                        auto cNode = dynamic_cast<ComputeNode*>(lNode->getNode(ixx));
                                        if (cNode->getOp() == ADD_OP)
                                        {
                                            cNode->getInput(1)->setName("t_iden_n");
                                            break;
                                        }
                                        ixx ++;
                                    }
                                }
                            } else if (cNode->getOp() == ROW_BROADCAST_OP)
                            {
                                inputName = cNode->getInput(1)->getName();
                            }
                        }
                        nodesMoved++;
                    }
                    ix++;
                }
                lNode->eraseFirstNLoopNodes(nodesMoved);
            }
            if (foundLeaning)
            {
                break;
            }
        }
    }

    // Complexity aware operator reordering
    // Reorder adjacent operations till no changes 
    static void complexityOperatorReordering (std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms,
        bool enableTim=false)
    {
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                bool changed = false;
                int numAggregations = 0;

                // If doing training invariant code motion, move the
                if (enableTim)
                {
                    // TODO leave alone for now. To be completed in the when implementing TIM
                     do
                    {
                        changed = false;

                        for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                        {
                            CIRNode* inNode = lNode->getNode(ix);
                            auto cNode = dynamic_cast<ComputeNode*>(inNode);
                            if (cNode->getOp() == FFN_OP)
                            {
                                DataNode* output = cNode->getOutput(0);
                                DataNode* input = cNode->getInput(0);

                                // Check next operation and get the complexity
                                if (lNode->getLoopNodeNum() <= (ix + 1))
                                {
                                    continue;
                                }

                                // Next node
                                // TODO Change later. Currently this is very brittle.
                                auto nextNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix + 1));
                                if (nextNode->getOp() == AGGREGATE_MUL_SUM_OP)
                                {
                                    changed = true;

                                    // Swap the nodes in the CIR
                                    lNode->swapNodes(ix, ix + 1);

                                    // Change the data for the DIR
                                    // The res input to the current FFN should be the res input to the next op
                                    auto nextInput = nextNode->getInput(0);
                                    auto nextOutput = nextNode->getOutput(0);

                                    // What you write to is going to change

                                    output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                    if (output->getName() == nextOutput->getName())
                                    {
                                        // New next node
                                        cNode->setInputDataNode(0, output);
                                        // New prev node
                                        nextNode->setInputDataNode(0, input);

                                        // New next node
                                        cNode->setOutputDataNode(0, nextOutput);
                                        // New prev node
                                        nextNode->setOutputDataNode(0, output);
                                    } else
                                    {
                                        // // New next node
                                        // nextOutput->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                        // New prev node
                                        nextNode->setInputDataNode(0, input);
                                    }
                                } else if (nextNode->getOp() == ROW_BROADCAST_OP)
                                {
                                    changed = true;

                                    // Swap the nodes in the CIR
                                    lNode->swapNodes(ix, ix + 1);

                                    // Change the data for the DIR
                                    // The res input to the current FFN should be the res input to the next op
                                    auto nextInput = nextNode->getInput(1); // These have different indices
                                    auto nextOutput = nextNode->getOutput(0);

                                    // What you write to is going to change
                                    // TODO -- should the inputs ansl change??
                                    output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                    // New next node
                                    cNode->setInputDataNode(0, output);
                                    cNode->setOutputDataNode(0, nextOutput);

                                    // New prev node
                                    nextNode->setInputDataNode(1, input);  // These have different indices
                                    nextNode->setOutputDataNode(0, output);
                                } else if (nextNode->getOp() == ADD_OP)
                                {
                                    changed = true;

                                    // Swap the nodes in the CIR
                                    lNode->swapNodes(ix, ix + 1);

                                    // Change the data for the DIR
                                    // The res input to the current FFN should be the res input to the next op
                                    auto nextInput0 = nextNode->getInput(0); // These have different indices
                                    auto nextInput1 = nextNode->getInput(1); // These have different indices
                                    auto nextOutput = nextNode->getOutput(0);

                                    // What you write to is going to change
                                    // The new prev node
                                    nextInput0->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                    nextInput1->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                    nextOutput->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                }   else if (nextNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
                                {
                                    changed = true;

                                    // Swap the nodes in the CIR
                                    lNode->swapNodes(ix, ix + 1);

                                    // Change the data for the DIR
                                    // The res input to the current FFN should be the res input to the next op
                                    auto nextInput = nextNode->getInput(0); // These have different indices
                                    auto nextOutput = nextNode->getOutput(0);

                                    // What you write to is going to change
                                    output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                    nextOutput->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                    // New next node
                                    cNode->setInputDataNode(0, output);
                                    // cNode->setOutputDataNode(0, nextOutput);

                                    // New prev node
                                    nextNode->setInputDataNode(0, input);  // These have different indices
                                    // nextNode->setOutputDataNode(0, output);
                                }
                            }
                        }
                    } while (changed);
                } else
                {
                    do
                    {
                        changed = false;

                        for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                        {
                            CIRNode* inNode = lNode->getNode(ix);
                            auto cNode = dynamic_cast<ComputeNode*>(inNode);
                            if (cNode->getOp() == FFN_OP)
                            {
                                DataNode* output = cNode->getOutput(0);
                                DataNode* input = cNode->getInput(0);

                                int inCols = cNode->getInput(1)->getDataInfo()->getDimRow();
                                int outCols = cNode->getInput(1)->getDataInfo()->getDimCol();

                                // If the output is larger than the input then move the weight update as far forward
                                // as possible
                                if (outCols > inCols)
                                {
                                    // Check next operation and get the complexity
                                    if (lNode->getLoopNodeNum() <= (ix + 1))
                                    {
                                        continue;
                                    }

                                    // Next node
                                    auto nextNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix + 1));
                                    if (nextNode->getOp() == AGGREGATE_MUL_SUM_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix + 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto nextInput = nextNode->getInput(0);
                                        auto nextOutput = nextNode->getOutput(0);

                                        output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                        if (output->getName() == nextOutput->getName())
                                        {
                                            // New next node
                                            cNode->setInputDataNode(0, output);
                                            // New prev node
                                            nextNode->setInputDataNode(0, input);

                                            // New next node
                                            cNode->setOutputDataNode(0, nextOutput);
                                            // New prev node
                                            nextNode->setOutputDataNode(0, output);
                                        } else
                                        {
                                            // New prev node
                                            nextNode->setInputDataNode(0, input);
                                        }
                                    } else if (nextNode->getOp() == ROW_BROADCAST_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix + 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto nextInput = nextNode->getInput(1);
                                        auto nextOutput = nextNode->getOutput(0);

                                        output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                        // New next node
                                        cNode->setInputDataNode(0, output);
                                        cNode->setOutputDataNode(0, nextOutput);

                                        // New prev node
                                        nextNode->setInputDataNode(1, input);
                                        nextNode->setOutputDataNode(0, output);
                                    }  else if (nextNode->getOp() == ADD_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix + 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto nextInput0 = nextNode->getInput(0); // These have different indices
                                        auto nextInput1 = nextNode->getInput(1); // These have different indices
                                        auto nextOutput = nextNode->getOutput(0);

                                        // What you write to is going to change
                                        // The new prev node
                                        nextInput0->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                        nextInput1->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                        nextOutput->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                    }   else if (nextNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix + 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto nextInput = nextNode->getInput(0); // These have different indices
                                        auto nextOutput = nextNode->getOutput(0);

                                        // What you write to is going to change
                                        output->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());
                                        nextOutput->getDataInfo()->setDims(-1, input->getDataInfo()->getDimCol());

                                        // New next node
                                        cNode->setInputDataNode(0, output);
                                        // cNode->setOutputDataNode(0, nextOutput);

                                        // New prev node
                                        nextNode->setInputDataNode(0, input);  // These have different indices
                                        // nextNode->setOutputDataNode(0, output);
                                    }
                                } else if (outCols < inCols)
                                {
                                    // Check next operation and get the complexity
                                    if (0 > (ix - 1))
                                    {
                                        continue;
                                    }

                                    // Next node
                                    auto prevNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix - 1));
                                    if (prevNode->getOp() == AGGREGATE_MUL_SUM_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix - 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto prevInput = prevNode->getInput(0);
                                        auto prevOutput = prevNode->getOutput(0);

                                        prevInput->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());

                                        if (output->getName() == prevOutput->getName())
                                        {
                                            // New prev node
                                            cNode->setInputDataNode(0, prevInput);
                                            // New next node
                                            prevNode->setInputDataNode(0, input);

                                            // New prev node
                                            cNode->setOutputDataNode(0, input);
                                            // New next node
                                            prevNode->setOutputDataNode(0, output);
                                        } else
                                        {
                                            prevNode->setInputDataNode(0, output);
                                        }
                                    } else if (prevNode->getOp() == ROW_BROADCAST_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix - 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto prevInput = prevNode->getInput(1);
                                        auto prevOutput = prevNode->getOutput(0);

                                        prevInput->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());

                                        // New prev node
                                        cNode->setInputDataNode(0, prevInput);
                                        cNode->setOutputDataNode(0, input);

                                        // New next node
                                        prevNode->setInputDataNode(1, input);
                                        prevNode->setOutputDataNode(0, output);
                                    }   else if (prevNode->getOp() == ADD_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix - 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto prevInput0 = prevNode->getInput(0);
                                        auto prevInput1 = prevNode->getInput(1);
                                        auto prevOutput = prevNode->getOutput(0);

                                        // What you write to is going to change
                                        // The new prev node
                                        prevInput0->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());
                                        prevInput1->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());
                                        prevOutput->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());
                                    }   else if (prevNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix - 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto prevInput = prevNode->getInput(0);
                                        auto prevOutput = prevNode->getOutput(0);

                                        prevInput->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());
                                        prevOutput->getDataInfo()->setDims(-1, output->getDataInfo()->getDimCol());

                                        // New prev node
                                        cNode->setInputDataNode(0, prevInput);
                                        // cNode->setOutputDataNode(0, input);

                                        // New next node
                                        prevNode->setInputDataNode(0, input);
                                        // prevNode->setOutputDataNode(0, output);
                                    }
                                }
                            }
                        }
                    } while (changed);
                }
            }
        }
    }
};

#endif //MIDDLE_END_H
