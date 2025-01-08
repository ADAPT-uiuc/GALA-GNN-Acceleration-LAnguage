#ifndef GNN_ACCELERATION_LANGUAGE_FRONTEND_IR
#define GNN_ACCELERATION_LANGUAGE_FRONTEND_IR

#include <string>
#include <vector>
#include <iostream>

class FrontendIRNode {
    public:
        std::string type;
        std::vector<std::string> params;
        std::vector<FrontendIRNode*> children;

        FrontendIRNode(std::string type) : type(type) {}
        ~FrontendIRNode(){
            for (auto child : children)
                    delete child;
        }

    /*
    Parameters for certain nodes
    i.e. "reddit" for a load node
         "rabbit" for a reorder node
    */
    void addParam(std::string param){
        params.push_back(param);
    }

    /*
    Nodes have children based on how the higher-level code gets parsed through the grammar
    The generateIR() function generates nodes for all children before generating for parent node
    */
    void addChild(FrontendIRNode* child){
        children.push_back(child);
    }
    /*
    Count the direct children from the root, used in the printParseTree() method
    */
    static int countDirectChildren(FrontendIRNode* root){
        int res = 0; 
        for (auto child : root->children)
            res++;

        return res;
    }

    static void printParseTree(FrontendIRNode* node, std::ostream& stream, std::vector<bool> flag, int depth = 0, bool isLast = false){
        /*
        stream << node->type << std::endl;
        if (!node->children.empty())
            stream << "children: " << std::endl;
        for (auto child : node->children){
            printParseTree(child, stream);
        }
        */
        if (node == NULL)
            return;

        for (int i = 1; i < depth; i++){
            if (flag[i]){
                std::cout << "| " << " " << " " << " ";
            }
            else{
                std::cout << " " << " " << " " << " ";
            }
        }
        if (depth == 0){
            std::cout << node->type << std::endl;
        }
        else if (isLast){
            std::cout << "+--- " << node->type << std::endl;
            flag[depth] = false;
        }
        else{
            std::cout << "+--- " << node->type << std::endl;
        }
        int it = 0;

        for (auto child : node->children){
            it++;
            printParseTree(child, stream, flag, depth+1, it == node->children.size()-1);
        }
        flag[depth] = true;

    }
    /*
    Recursively generate code from the parse tree (parse through tree "left to right")
    Initial call takes root node and output stream
    */
    static void generateIR(FrontendIRNode* node, std::ostream& stream){
        if (node->type == "dsl_prog"){
            // stream << "-- init stuff not associated with specific code --" << std::endl;
            stream << "// Init point counter" << std::endl;
            stream << "std::vector<ComputeNode*> program;" << std::endl;
            stream << "auto pc = PointCounter();" << std::endl;
            stream << "int now;" << std::endl;
        }

        if (node->type == "load"){
            // stream << "-- loading graph lines with dataset and features --" << std::endl;

            stream << "// 1 - Load graph dataset with data nodes for the graph and features" << std::endl;
            stream << "now = pc.getPoint();" << std::endl;
            stream << "auto loadCompute = StatementNode(LOAD_OP, now);" << std::endl;
            stream << "loadCompute.addParam(\" << node->param[0] << \");" << std::endl;
            stream << "// Graph" << std::endl;
            stream << "auto initialGraph = DataList(CSR_STYPE, true);" << std::endl;
            stream << "auto graphData = DataNode(\"gGraph\", UINT32, UINT64, F32, now, &initialGraph);" << std::endl;
            stream << "// Feat" << std::endl;
            stream << "auto initialFeat = DataList(RM_DTYPE, true);" << std::endl;
            stream << "auto featData = DataNode(\"gFeat\", UINT32, UINT64, F32, now, &initialFeat);" << std::endl;
            stream << "// Relation -- TODO Ignore relations for now. Use the visitor class design" << std::endl;
            stream << "auto graphFeatRel = RelationEdge(&graphData, ROW_RELATION, &featData, ROW_RELATION);" << std::endl;
            stream << "loadCompute.addOutputData(&graphData);" << std::endl;
            stream << "loadCompute.addOutputData(&featData);" << std::endl;

        }
        else if (node->type == "ReorderGraph"){
            // stream << "-- graph transfomrations (reordering) --" << std::endl;
            stream << "// 2 - Graph Transformations" << std::endl;
            stream << "now = pc.getPoint();" << std::endl;
            stream << "auto reorderGraph = TransformData(REORDER_TRNS);" << std::endl;
            stream << "reorderGraph.addParam(\"" << node->params[0] << "\");" << std::endl;
        }   
        else if (node->type == "Layer"){

        }
        else if (node->type == "Model"){

        }
        else if (node->type == "Train"){

        }
        else if (node->type == "comment"){
            node->params[0].erase(node->params[0].size() - 1);
            stream << node->params[0] << " (user comment)" << std::endl;
        }

        for (auto child : node->children){
            generateIR(child, stream);
        }
    }
};

#endif // GNN_ACCELERATION_LANGUAGE_FRONTEND_IR