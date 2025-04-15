//
// Created by damitha on 4/15/25.
//

#ifndef CONTEXT_H
#define CONTEXT_H

#include <vector>
#include <map>
#include "frontendIR.h"

extern std::vector<CIRNode*>* program;
extern std::vector<RelationEdge*>* dependencies;
extern std::vector<RelationEdge*>* associations;
extern std::vector<TransformEdge*>* transforms;

#endif //CONTEXT_H
