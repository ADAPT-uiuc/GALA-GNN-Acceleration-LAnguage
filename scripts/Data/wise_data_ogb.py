import os
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset
import dgl

basedir = "../Environments/WiseGraph/data/"
datasets = ["ogbn-arxiv", "ogbn-products"]
undirected = True  # Change to False if you want the directed version

for name in datasets:
    print(f"Processing dataset: {name}")

    # Load the dataset
    dataset = DglNodePropPredDataset(name=name, root="../../Data/ogb/")
    split_idx = dataset.get_idx_split()  # Get the split indices for train/validation/test
    dgl_graph, labels = dataset[0]  # Get the DGL graph and labels
    
    dgl_graph = dgl.remove_self_loop(dgl_graph)
    dgl_graph = dgl.add_self_loop(dgl_graph)

    # Convert DGL graph to scipy sparse CSR format
    num_nodes = dgl_graph.num_nodes()
    edges = dgl_graph.edges()
    adj_csr = sp.csr_matrix((np.ones(edges[0].shape[0]), (edges[0].numpy(), edges[1].numpy())), shape=(num_nodes, num_nodes))

    # Convert to undirected if specified
    if undirected:
        adj_csr = adj_csr + adj_csr.T - sp.diags(adj_csr.diagonal())

    # Debugging prints
    print(f"NumNodes: {num_nodes}")
    print(f"CSR matrix shape: {adj_csr.shape}")
    print(f"Length of indptr (ptr): {len(adj_csr.indptr)}")  # Should be num_nodes + 1
    print(f"Length of indices (idx): {len(adj_csr.indices)}")  # Should match number of edges

    # Define file paths for saving
    dataset_dir = f"{basedir}{name}/processed/"
    os.makedirs(dataset_dir, exist_ok=True)
    ptr_file = f"{dataset_dir}csr_ptr_{'undirected' if undirected else 'directed'}.dat"
    idx_file = f"{dataset_dir}csr_idx_{'undirected' if undirected else 'directed'}.dat"
    num_nodes_file = f"{dataset_dir}num_nodes.txt"
    feature_file = f"{dataset_dir}node_features.dat"
    label_file = f"{dataset_dir}node_labels.dat"
    train_idx_file = f"{dataset_dir}train_idx.dat"
    train_mask_file = f"{dataset_dir}train_mask.dat"
    test_idx_file = f"{dataset_dir}test_idx.dat"
    test_mask_file = f"{dataset_dir}test_mask.dat"
    val_idx_file = f"{dataset_dir}val_idx.dat"
    val_mask_file = f"{dataset_dir}val_mask.dat"

    # Save the CSR matrix in the specified format using .tofile
    adj_csr.indptr.astype(np.int64).tofile(ptr_file)
    adj_csr.indices.astype(np.int64).tofile(idx_file)
    print(f"Pointer file saved as '{ptr_file}'")
    print(f"Index file saved as '{idx_file}'")

    # Save the number of nodes to num_nodes.txt
    with open(num_nodes_file, "w") as f:
        f.write(str(num_nodes))
    print(f"Number of nodes saved as '{num_nodes_file}'")

    # Save node features
    features = dgl_graph.ndata['feat'].numpy()
    features.astype(np.float32).tofile(feature_file)
    print(f"Node features saved to '{feature_file}'")

    # Save node labels
    labels.numpy().astype(np.int64).tofile(label_file)
    print(f"Node labels saved to '{label_file}'")

    # Save training, validation, and test indices and masks
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    # Create masks for train, validation, and test sets
    train_mask = np.zeros(num_nodes, dtype=np.int8)
    val_mask = np.zeros(num_nodes, dtype=np.int8)
    test_mask = np.zeros(num_nodes, dtype=np.int8)

    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    # Save indices and masks
    train_idx.numpy().astype(np.int64).tofile(train_idx_file)
    train_mask.tofile(train_mask_file)
    print(f"Training indices saved to '{train_idx_file}'")
    print(f"Training mask saved to '{train_mask_file}'")

    val_idx.numpy().astype(np.int64).tofile(val_idx_file)
    val_mask.tofile(val_mask_file)
    print(f"Validation indices saved to '{val_idx_file}'")
    print(f"Validation mask saved to '{val_mask_file}'")

    test_idx.numpy().astype(np.int64).tofile(test_idx_file)
    test_mask.tofile(test_mask_file)
    print(f"Test indices saved to '{test_idx_file}'")
    print(f"Test mask saved to '{test_mask_file}'")

    # ========= Loading and Validation Section =========

    # Define properties and paths for loading
    prop = "undirected" if undirected else "directed"
    ptr_path = f"{dataset_dir}csr_ptr_{prop}.dat"
    idx_path = f"{dataset_dir}csr_idx_{prop}.dat"
    num_nodes_path = f"{dataset_dir}num_nodes.txt"
    feature_path = f"{dataset_dir}node_features.dat"
    label_path = f"{dataset_dir}node_labels.dat"
    train_idx_path = f"{dataset_dir}train_idx.dat"
    train_mask_path = f"{dataset_dir}train_mask.dat"
    test_idx_path = f"{dataset_dir}test_idx.dat"
    test_mask_path = f"{dataset_dir}test_mask.dat"
    val_idx_path = f"{dataset_dir}val_idx.dat"
    val_mask_path = f"{dataset_dir}val_mask.dat"

    # Load the CSR data using np.fromfile
    ptr = np.fromfile(ptr_path, dtype=np.int64)
    idx = np.fromfile(idx_path, dtype=np.int64)

    # Load number of nodes and validate it
    with open(num_nodes_path, "r") as f:
        loaded_num_nodes = int(f.read().strip())

    # Load node features for validation
    loaded_features = np.fromfile(feature_path, dtype=np.float32).reshape(num_nodes, -1)

    # Load labels for validation
    loaded_labels = np.fromfile(label_path, dtype=np.int64).reshape(-1, 1)

    # Load training, validation, and test indices and masks for validation
    loaded_train_idx = np.fromfile(train_idx_path, dtype=np.int64)
    loaded_train_mask = np.fromfile(train_mask_path, dtype=np.int8)
    loaded_test_idx = np.fromfile(test_idx_path, dtype=np.int64)
    loaded_test_mask = np.fromfile(test_mask_path, dtype=np.int8)
    loaded_val_idx = np.fromfile(val_idx_path, dtype=np.int64)
    loaded_val_mask = np.fromfile(val_mask_path, dtype=np.int8)

    print("\n=== Validation Results ===")
    print(f"Number of nodes loaded: {loaded_num_nodes}")
    print(f"Pointer array (ptr) length: {len(ptr)}")
    print(f"Index array (idx) length: {len(idx)}")
    print(f"Node features shape: {loaded_features.shape}")
    print(f"Node labels shape: {loaded_labels.shape}")
    print(f"Training indices length: {len(loaded_train_idx)}")
    print(f"Training mask shape: {loaded_train_mask.shape}")
    print(f"Test indices length: {len(loaded_test_idx)}")
    print(f"Test mask shape: {loaded_test_mask.shape}")
    print(f"Validation indices length: {len(loaded_val_idx)}")
    print(f"Validation mask shape: {loaded_val_mask.shape}")

    # Validation checks
    assert loaded_num_nodes == num_nodes, "Number of nodes does not match the original graph."
    assert len(ptr) == num_nodes + 1, "Pointer array length should be num_nodes + 1."
    assert ptr[-1] == len(idx), "Last value of ptr should match length of idx (total non-zeros)."
    assert loaded_features.shape[0] == num_nodes, "Feature array length should match the number of nodes."
    assert loaded_labels.shape[0] == num_nodes, "Label array length should match the number of nodes."
    assert np.array_equal(loaded_train_idx, train_idx), "Training indices do not match."
    assert np.array_equal(loaded_train_mask, train_mask), "Training mask does not match."
    assert np.array_equal(loaded_test_idx, test_idx), "Test indices do not match."
    assert np.array_equal(loaded_test_mask, test_mask), "Test mask does not match."
    assert np.array_equal(loaded_val_idx, val_idx), "Validation indices do not match."
    assert np.array_equal(loaded_val_mask, val_mask), "Validation mask does not match."

    print("Validation successful: Loaded CSR structure, features, labels, and train/test/val data match the saved data.")

print("Processing and validation complete for all datasets.")
