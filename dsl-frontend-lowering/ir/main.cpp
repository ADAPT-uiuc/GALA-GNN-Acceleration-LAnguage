#include <iostream>
#include "frontendIR.h"

int main(){
    FrontendIRNode* root = new FrontendIRNode("Root");
    FrontendIRNode* load = new FrontendIRNode("LoadGraph");
    load->addParam("Reddit");

    FrontendIRNode* reorderRabbit = new FrontendIRNode("ReorderGraph");
    reorderRabbit->addParam("rabbit");

    root->addChild(load);
    root->addChild(reorderRabbit);



    FrontendIRNode::generateIR(root, std::cout);
    delete root;

    
    return 0;
}