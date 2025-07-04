import os
import numpy as np
import scipy.sparse as sp
from dgl.data import RedditDataset, CoraGraphDataset, PubmedGraphDataset, CoraFullDataset
import dgl
from dgl import AddSelfLoop
import torch

basedir = "../Environments/WiseGraph/data/" 
os.makedirs(basedir, exist_ok=True)
undirected = True  # Change to False if you want the directed version

# Datasets to process
datasets = {
    "reddit": RedditDataset,
    "cora": CoraGraphDataset,
    "pubmed": PubmedGraphDataset,
    "corafull": CoraFullDataset
}

for dataset_name, dataset_class in datasets.items():
    print(f"\nProcessing {dataset_name} dataset")

    # Load the dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    dataset = dataset_class(transform=transform)
    graph = dataset[0]  # Get the DGL graph
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    # Get the adjacency matrix in CSR format
    src, dst = graph.edges()
    edges = np.stack((src.numpy(), dst.numpy()), axis=0)
    adj_csr = sp.csr_matrix((np.ones(num_edges), (edges[0], edges[1])), shape=(num_nodes, num_nodes))

    # Convert to undirected if specified
    if undirected:
        adj_csr = adj_csr + adj_csr.T - sp.diags(adj_csr.diagonal())

    # Define file paths for saving
    dataset_dir = f"{basedir}{dataset_name}/processed/"
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
    features = graph.ndata['feat'].numpy()  # Access features
    features.astype(np.float32).tofile(feature_file)
    print(f"Node features saved to '{feature_file}'")

    # Save node labels
    labels = graph.ndata['label'].numpy()  # Access labels
    labels.astype(np.int64).tofile(label_file)
    print(f"Node labels saved to '{label_file}'")

    # Handle missing masks in datasets like corafull
    if 'train_mask' not in graph.ndata:
        np.random.seed(42)  # For reproducibility
        n_nodes = graph.num_nodes()
        indices = np.arange(n_nodes)
        np.random.shuffle(indices)
        train_end = int(0.6 * n_nodes)
        val_end = int(0.8 * n_nodes)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = np.zeros(n_nodes, dtype=bool)
        val_mask = np.zeros(n_nodes, dtype=bool)
        test_mask = np.zeros(n_nodes, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        graph.ndata['train_mask'] = torch.from_numpy(train_mask)
        graph.ndata['val_mask'] = torch.from_numpy(val_mask)
        graph.ndata['test_mask'] = torch.from_numpy(test_mask)

    # Save training indices and mask
    train_mask = graph.ndata['train_mask'].numpy()
    train_idx = np.where(train_mask)[0]
    train_idx.astype(np.int64).tofile(train_idx_file)
    train_mask.astype(np.int8).tofile(train_mask_file)
    print(f"Training indices saved to '{train_idx_file}'")
    print(f"Training mask saved to '{train_mask_file}'")

    # Save test indices and mask
    test_mask = graph.ndata['test_mask'].numpy()
    test_idx = np.where(test_mask)[0]
    test_idx.astype(np.int64).tofile(test_idx_file)
    test_mask.astype(np.int8).tofile(test_mask_file)
    print(f"Test indices saved to '{test_idx_file}'")
    print(f"Test mask saved to '{test_mask_file}'")

    # Save validation indices and mask
    val_mask = graph.ndata['val_mask'].numpy()
    val_idx = np.where(val_mask)[0]
    val_idx.astype(np.int64).tofile(val_idx_file)
    val_mask.astype(np.int8).tofile(val_mask_file)
    print(f"Validation indices saved to '{val_idx_file}'")
    print(f"Validation mask saved to '{val_mask_file}'")

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

    # Load training, test, and validation indices and masks for validation
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

    print(f"Validation successful: Data for the {dataset_name} dataset matches the saved data.\n")
