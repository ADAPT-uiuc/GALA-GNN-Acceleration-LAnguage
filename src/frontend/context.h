//
// Created by damitha on 4/15/25.
//

#ifndef CONTEXT_H
#define CONTEXT_H

#include "../ir/data.h"
#include "../ir/compute.h"
#include "../ir/frontend_metadata.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>


class GALAFEContext {
    public:
        static std::vector<CIRNode*> program;
        static std::vector<RelationEdge*> dependencies;
        static std::vector<RelationEdge*> associations;
        static std::vector<TransformEdge*> transforms;

		static bool operator_reordering;
		static bool sparse_rewrites;
		static bool training_subgraph;
		static bool train_code_motion;
};

#endif //CONTEXT_H
