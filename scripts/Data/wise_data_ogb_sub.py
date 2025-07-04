import os
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset
import dgl

basedir = "../Environments/WiseGraph/data/"
dataset_name = "ogbn-papers100M"

# Define the percentages (as fractions) for which subgraphs will be saved
percentages = [0.01, 0.02, 0.05, 0.10]
undirected = True  # set to False for a directed version

print(f"Loading dataset: {dataset_name}")
# Load the dataset
dataset = DglNodePropPredDataset(name=dataset_name)
split_idx = dataset.get_idx_split()  # train/valid/test split indices
dgl_graph, labels = dataset[0]         # get the DGL graph and node labels

# Preprocess full graph: remove self-loops then add self-loops
dgl_graph = dgl.remove_self_loop(dgl_graph)
dgl_graph = dgl.add_self_loop(dgl_graph)

total_num_nodes = dgl_graph.num_nodes()
print(f"Total number of nodes in the original graph: {total_num_nodes}")

# Convert the original split indices to numpy arrays for easier filtering
train_idx_all = split_idx['train'].numpy()
val_idx_all = split_idx['valid'].numpy()
test_idx_all = split_idx['test'].numpy()

# Loop over each percentage to create, save, and validate a subgraph
for perc in percentages:
    print(f"\n=== Processing subgraph with {int(perc*100)}% of nodes ===")
    num_nodes_sub = int(total_num_nodes * perc)
    
    # Select the first num_nodes_sub nodes (assuming original ordering)
    nodes = np.arange(num_nodes_sub)
    
    # Extract the subgraph; note that dgl.node_subgraph reindexes nodes.
    subgraph = dgl.node_subgraph(dgl_graph, nodes)
    # Retrieve the original node IDs stored in the subgraph
    orig_nids = subgraph.ndata[dgl.NID].numpy()
    # Build a mapping from original node id to new subgraph node id
    mapping = {orig: new for new, orig in enumerate(orig_nids)}
    
    # Subset node labels and features using the original node ids
    sub_labels = labels[orig_nids]
    sub_features = subgraph.ndata['feat'].numpy()
    
    # Filter original split indices to include only nodes in the subgraph,
    # then remap these indices to the subgraph’s ordering.
    train_idx_sub_orig = train_idx_all[np.isin(train_idx_all, orig_nids)]
    val_idx_sub_orig = val_idx_all[np.isin(val_idx_all, orig_nids)]
    test_idx_sub_orig = test_idx_all[np.isin(test_idx_all, orig_nids)]
    
    train_idx_sub = np.array([mapping[i] for i in train_idx_sub_orig])
    val_idx_sub = np.array([mapping[i] for i in val_idx_sub_orig])
    test_idx_sub = np.array([mapping[i] for i in test_idx_sub_orig])
    
    # Create boolean masks for the subgraph splits
    sub_num_nodes = subgraph.num_nodes()
    train_mask = np.zeros(sub_num_nodes, dtype=np.int8)
    val_mask = np.zeros(sub_num_nodes, dtype=np.int8)
    test_mask = np.zeros(sub_num_nodes, dtype=np.int8)
    train_mask[train_idx_sub] = 1
    val_mask[val_idx_sub] = 1
    test_mask[test_idx_sub] = 1
    
    # (Optional) Reprocess the subgraph: remove then add self-loops
    subgraph = dgl.remove_self_loop(subgraph)
    subgraph = dgl.add_self_loop(subgraph)
    
    # Convert the subgraph to a scipy sparse CSR matrix.
    edges = subgraph.edges()
    adj_csr = sp.csr_matrix(
        (np.ones(edges[0].shape[0]), (edges[0].numpy(), edges[1].numpy())),
        shape=(sub_num_nodes, sub_num_nodes)
    )
    
    # Convert to undirected if specified
    if undirected:
        adj_csr = adj_csr + adj_csr.T - sp.diags(adj_csr.diagonal())
    
    # Define directory paths for saving the subgraph data.
    sub_dir = f"{basedir}{dataset_name}_{int(perc*100)}/processed/"
    os.makedirs(sub_dir, exist_ok=True)
    prop = 'undirected' if undirected else 'directed'
    ptr_file = f"{sub_dir}csr_ptr_{prop}.dat"
    idx_file = f"{sub_dir}csr_idx_{prop}.dat"
    num_nodes_file = f"{sub_dir}num_nodes.txt"
    feature_file = f"{sub_dir}node_features.dat"
    label_file = f"{sub_dir}node_labels.dat"
    train_idx_file = f"{sub_dir}train_idx.dat"
    train_mask_file = f"{sub_dir}train_mask.dat"
    test_idx_file = f"{sub_dir}test_idx.dat"
    test_mask_file = f"{sub_dir}test_mask.dat"
    val_idx_file = f"{sub_dir}val_idx.dat"
    val_mask_file = f"{sub_dir}val_mask.dat"
    
    # Save the CSR matrix components using .tofile.
    adj_csr.indptr.astype(np.int64).tofile(ptr_file)
    adj_csr.indices.astype(np.int64).tofile(idx_file)
    print(f"Subgraph: CSR pointer file saved as '{ptr_file}'")
    print(f"Subgraph: CSR index file saved as '{idx_file}'")
    
    # Save the number of nodes.
    with open(num_nodes_file, "w") as f:
        f.write(str(sub_num_nodes))
    print(f"Subgraph: Number of nodes saved as '{num_nodes_file}'")
    
    # Save node features.
    sub_features.astype(np.float32).tofile(feature_file)
    print(f"Subgraph: Node features saved to '{feature_file}'")
    
    # Save node labels.
    sub_labels.numpy().astype(np.int64).tofile(label_file)
    print(f"Subgraph: Node labels saved to '{label_file}'")
    
    sub_labels_np = sub_labels.numpy().flatten()
    print("Unique labels in subgraph:", np.unique(sub_labels_np))
    print("Min label:", sub_labels_np.min(), "Max label:", sub_labels_np.max())
    
    # Save train/validation/test indices and masks.
    train_idx_sub.astype(np.int64).tofile(train_idx_file)
    train_mask.tofile(train_mask_file)
    print(f"Subgraph: Training indices saved to '{train_idx_file}'")
    print(f"Subgraph: Training mask saved to '{train_mask_file}'")
    
    val_idx_sub.astype(np.int64).tofile(val_idx_file)
    val_mask.tofile(val_mask_file)
    print(f"Subgraph: Validation indices saved to '{val_idx_file}'")
    print(f"Subgraph: Validation mask saved to '{val_mask_file}'")
    
    test_idx_sub.astype(np.int64).tofile(test_idx_file)
    test_mask.tofile(test_mask_file)
    print(f"Subgraph: Test indices saved to '{test_idx_file}'")
    print(f"Subgraph: Test mask saved to '{test_mask_file}'")
    
    # ========= Validation Section for the Subgraph =========
    # Define paths for reloading the saved data.
    prop = 'undirected' if undirected else 'directed'
    ptr_path = f"{sub_dir}csr_ptr_{prop}.dat"
    idx_path = f"{sub_dir}csr_idx_{prop}.dat"
    num_nodes_path = f"{sub_dir}num_nodes.txt"
    feature_path = f"{sub_dir}node_features.dat"
    label_path = f"{sub_dir}node_labels.dat"
    train_idx_path = f"{sub_dir}train_idx.dat"
    train_mask_path = f"{sub_dir}train_mask.dat"
    test_idx_path = f"{sub_dir}test_idx.dat"
    test_mask_path = f"{sub_dir}test_mask.dat"
    val_idx_path = f"{sub_dir}val_idx.dat"
    val_mask_path = f"{sub_dir}val_mask.dat"
    
    # Reload the CSR structure.
    loaded_ptr = np.fromfile(ptr_path, dtype=np.int64)
    loaded_idx = np.fromfile(idx_path, dtype=np.int64)
    
    # Reload number of nodes.
    with open(num_nodes_path, "r") as f:
        loaded_num_nodes = int(f.read().strip())
    assert loaded_num_nodes == sub_num_nodes, "Mismatch in number of nodes."
    
    # Reload node features.
    loaded_features = np.fromfile(feature_path, dtype=np.float32).reshape(sub_num_nodes, -1)
    assert loaded_features.shape[0] == sub_num_nodes, "Mismatch in node features shape."
    
    # Reload node labels.
    loaded_labels = np.fromfile(label_path, dtype=np.int64).reshape(-1, 1)
    assert loaded_labels.shape[0] == sub_num_nodes, "Mismatch in node labels shape."
    
    # Reload split indices and masks.
    loaded_train_idx = np.fromfile(train_idx_path, dtype=np.int64)
    loaded_train_mask = np.fromfile(train_mask_path, dtype=np.int8)
    loaded_val_idx = np.fromfile(val_idx_path, dtype=np.int64)
    loaded_val_mask = np.fromfile(val_mask_path, dtype=np.int8)
    loaded_test_idx = np.fromfile(test_idx_path, dtype=np.int64)
    loaded_test_mask = np.fromfile(test_mask_path, dtype=np.int8)
    
    # Validate that the saved splits match the original subgraph splits.
    assert np.array_equal(loaded_train_idx, train_idx_sub), "Mismatch in training indices."
    assert np.array_equal(loaded_train_mask, train_mask), "Mismatch in training mask."
    assert np.array_equal(loaded_val_idx, val_idx_sub), "Mismatch in validation indices."
    assert np.array_equal(loaded_val_mask, val_mask), "Mismatch in validation mask."
    assert np.array_equal(loaded_test_idx, test_idx_sub), "Mismatch in test indices."
    assert np.array_equal(loaded_test_mask, test_mask), "Mismatch in test mask."
    
    print(f"Validation successful for subgraph with {int(perc*100)}% nodes.")

print("\nProcessing and validation complete for all subgraph percentages.")
