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

    static int getEdgeIndex(std::vector<DataEdge*> edges, DataNode *n1, DataNode *n2)
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
    }

    static void trainingSubGraph(std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        // Iterate through program to locate the training loop
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto lNode = dynamic_cast<TrainingLoopNode*>(outNode);
            // Do this transformation only if you have
            if (lNode)
            {
                int countAggregations = 0;

                // TODO Create the initial input graph.
                //  i.e. add computations to the IR to create these results

                for (int ix = loopNode->getLoopNodeNum() - 1; ix >= 0; ix--)
                {
                    // TODO Change this to a stringed set of aggregations not just aggregation
                    //  i.e. result of aggregation 1 is used by aggregation 2
                    //  if the aggregations are separate then they can still be independent
                    CIRNode* inNode = loopNode->getNode(ix);
                    if (cNode->getOpType() == AGGREGATE_NODE)
                    {
                        countAggregations++;
                        // TODO create the new graph
                        //  i.e. add computations to the IR to create these results

                    } else
                    {
                        // TODO Use the most recent input graph as a graph input if graph is used
                        // Use previous
                    }
                }
            }
        }
    }

    // TODO Sparsity aware re-write
    static void sparsityAwareRewrites(std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {

    }

    // TODO Training invariant code motion
    static void trainingInvariantCodeMotion (std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {

    }

    // TODO Complexity aware operator reordering
    // Reorder adjacent operations till no changes 
    static void ComplexityOperatorReordering (std::vector<CIRNode*>& program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms,
        bool enableTim)
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
                } else
                {
                    do
                    {
                        for (int ix = 0; ix < lNode->getLoopNodeNum(); ix++)
                        {
                            CIRNode* inNode = lNode->getNode(ix);
                            auto cNode = dynamic_cast<ComputeNode*>(inNode);
                            if (cNode->getOp() == FFN_OP)
                            {
                                DataNode* output = cNode->getOutput(0);
                                DataNode* input = cNode->getInput(0);

                                // If the output is larger than the input then move as far head as possible
                                if (output->getDataInfo()->getDimCol() > input->getDataInfo()->getDimCol())
                                {
                                    // Check next operation and get the complexity
                                    if (lNode->getLoopNodeNum() <= (ix + 1))
                                    {
                                        continue;
                                    }

                                    // Next node
                                    auto nextNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix + 1));
                                    if (nextNode->getOP() == AGGREGATE_MUL_SUM_OP ||
                                        nextNode->getOP() == ROW_BROADCAST_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix + 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto nextInput = nextNode->getInput(0);
                                        auto nextOutput = nextNode->getOutput(0);
                                        cNode->setInputDataNode(0, nextInput);
                                        nextNode->setInputDataNode(0, input);

                                        // Change the dependency graph
                                        auto inOutDepIndex = getEdgeIndex(dependencies, input, nextInput);
                                        dependencies.erase(dependencies.begin() + inOutDepIndex);
                                        auto updatedDependency = RelationEdge(nextInput, ALL_RELATION, input, ALL_RELATION);
                                        dependencies.push_back(&updatedDependency);
                                        // Swap dependency for new output
                                        auto outResDepIndex = getEdgeIndex(dependencies, nextInput, nextOutput);
                                        dependencies.erase(dependencies.begin() + outResDepIndex);
                                        auto updatedResDependency = RelationEdge(input, ALL_RELATION, nextOutput, ALL_RELATION);
                                        dependencies.push_back(&updatedResDependency);
                                    }
                                } else if (output->getDataInfo()->getDimCol() < input->getDataInfo()->getDimCol())
                                {
                                    // Check next operation and get the complexity
                                    if (0 > (ix - 1))
                                    {
                                        continue;
                                    }

                                    // Next node
                                    auto prevNode = dynamic_cast<ComputeNode*>(lNode->getNode(ix - 1));
                                    if (prevNode->getOP() == AGGREGATE_MUL_SUM_OP ||
                                        prevNode->getOP() == ROW_BROADCAST_OP)
                                    {
                                        changed = true;

                                        // Swap the nodes in the CIR
                                        lNode->swapNodes(ix, ix - 1);

                                        // Change the data for the DIR
                                        // The res input to the current FFN should be the res input to the next op
                                        auto prevInput = prevNode->getInput(0);
                                        auto prevOutput = prevNode->getOutput(0);
                                        cNode->setInputDataNode(0, prevInput);
                                        prevNode->setInputDataNode(0, input);

                                        // Change the dependency graph
                                        auto inOutDepIndex = getEdgeIndex(dependencies, prevInput, input);
                                        dependencies.erase(dependencies.begin() + inOutDepIndex);
                                        auto updatedDependency = RelationEdge(input, ALL_RELATION, prevInput, ALL_RELATION);
                                        dependencies.push_back(&updatedDependency);
                                        // Swap dependency for new output
                                        auto outResDepIndex = getEdgeIndex(dependencies, prevInput, prevOutput);
                                        dependencies.erase(dependencies.begin() + outResDepIndex);
                                        auto updatedResDependency = RelationEdge(input, ALL_RELATION, prevOutput, ALL_RELATION);
                                        dependencies.push_back(&updatedResDependency);
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
