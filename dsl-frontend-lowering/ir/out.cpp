// Init point counter
std::vector<ComputeNode*> program;
auto pc = PointCounter();
int now;
// 1 - Load graph dataset with data nodes for the graph and features
now = pc.getPoint();
auto loadCompute = StatementNode(LOAD_OP, now);
loadCompute.addParam(" << node->param[0] << ");
// Graph
auto initialGraph = DataList(CSR_STYPE, true);
auto graphData = DataNode("gGraph", UINT32, UINT64, F32, now, &initialGraph);
// Feat
auto initialFeat = DataList(RM_DTYPE, true);
auto featData = DataNode("gFeat", UINT32, UINT64, F32, now, &initialFeat);
// Relation -- TODO Ignore relations for now. Use the visitor class design
auto graphFeatRel = RelationEdge(&graphData, ROW_RELATION, &featData, ROW_RELATION);
loadCompute.addOutputData(&graphData);
loadCompute.addOutputData(&featData);
// 2 - Graph Transformations
now = pc.getPoint();
auto reorderGraph = TransformData(REORDER_TRNS);
reorderGraph.addParam("rabbit");
