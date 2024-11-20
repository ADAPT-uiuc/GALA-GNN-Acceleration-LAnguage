echo "****GCN*****" > fwd_gala.txt
########## Actual ###################
echo "CoraGraphDataset" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 10 >> fwd_gala.txt
echo "PubmedGraphDataset" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 10 >> fwd_gala.txt
echo "CoraFullDataset" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 10 >> fwd_gala.txt
echo "RedditDataset" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/RedditDataset/ 100 37000 10 >> fwd_gala.txt
echo "ogbn-arxiv" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 25000 10 >> fwd_gala.txt
echo "ogbn-products" >> fwd_gala.txt
build/gala_tile_tir_op /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 10 >> fwd_gala.txt
echo "--------sparse--------" >> fwd_gala.txt
echo "CoraGraphDataset" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 10 >> fwd_gala.txt
echo "PubmedGraphDataset" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 10 >> fwd_gala.txt
echo "CoraFullDataset" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 10 >> fwd_gala.txt
echo "RedditDataset" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/RedditDataset/ 100 37000 10 >> fwd_gala.txt
echo "ogbn-arxiv" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 25000 10 >> fwd_gala.txt
echo "ogbn-products" >> fwd_gala.txt
build/gala_tile_tir_op_sparse /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 10 >> fwd_gala.txt
echo "****GAT*****" >> fwd_gala.txt
echo "CoraGraphDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 10 >> fwd_gala.txt
echo "PubmedGraphDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 10 >> fwd_gala.txt
echo "CoraFullDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 10 >> fwd_gala.txt
echo "RedditDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/RedditDataset/ 100 37000 10 >> fwd_gala.txt
echo "ogbn-arxiv" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 25000 10 >> fwd_gala.txt
echo "ogbn-products" >> fwd_gala.txt
build/gala_gat_tile_tir_op /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 10 >> fwd_gala.txt
echo "--------sparse--------" >> fwd_gala.txt
echo "CoraGraphDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 10 >> fwd_gala.txt
echo "PubmedGraphDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 10 >> fwd_gala.txt
echo "CoraFullDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 10 >> fwd_gala.txt
echo "RedditDataset" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/RedditDataset/ 100 37000 10 >> fwd_gala.txt
echo "ogbn-arxiv" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 25000 10 >> fwd_gala.txt
echo "ogbn-products" >> fwd_gala.txt
build/gala_gat_tile_tir_op_sparse /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 10 >> fwd_gala.txt
echo "****GIN*****" >> fwd_gala.txt
echo "CoraGraphDataset" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 10 >> fwd_gala.txt
echo "PubmedGraphDataset" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 10 >> fwd_gala.txt
echo "CoraFullDataset" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 10 >> fwd_gala.txt
echo "RedditDataset" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/RedditDataset/ 100 37000 10 >> fwd_gala.txt
echo "ogbn-arxiv" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 25000 10 >> fwd_gala.txt
echo "ogbn-products" >> fwd_gala.txt
build/gala_gin_tile_tir_op /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 10 >> fwd_gala.txt

