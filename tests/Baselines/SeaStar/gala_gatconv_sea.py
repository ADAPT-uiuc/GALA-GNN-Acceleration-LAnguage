"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
# from dgl.nn.functional import edge_softmax
import timeit

# Seastar
import dgl.backend as B
import dgl
from seastar import CtxManager


# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


# pylint: enable=W0235
class TimedGATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        dont_recompute=True,
        profile_spmm=False,
        discard_k=1
    ):
        super(TimedGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._dont_recompute = dont_recompute
        self._profile_spmm = profile_spmm
        if profile_spmm:
            self.discard_k = discard_k
            self.needs_lazy_update=True
            self._time_stats = {"SpMM":0.0,"GeMM":0.0,"call_count":0}

        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False
        )
        self.attn_l = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope
        if bias:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
        else:
            self.register_buffer("bias", None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.cm = CtxManager(dgl.backend.run_egl)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        if self._profile_spmm:
            self._time_stats = {"SpMM":0.0,"GeMM":0.0,"call_count":0}
            self.needs_lazy_update=True

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def reset_timers(self):
        self._time_stats["SpMM"] = 0.0
        self._time_stats["GeMM"]=0.0

    def update_timer(self,key:str,tn=0.0):
        
        if key=="SpMM" or key=="GeMM":
            # Maintain a running sum
            an = self._time_stats[key]
            n = self._time_stats["call_count"]

            if n>=0:
                self._time_stats[key] = (an + tn) 
            else:
                raise ValueError('Invalid value for call count')

        elif key=="call_count":
            self._time_stats[key] = self._time_stats[key] + 1

    def get_time_stats(self):
        if not self._profile_spmm:
            raise DGLError( 'Time logging is disabled for SpMM. Please'
                            ' re-define the layer with timing enabled')
        
        # Lazily update time to track average
        if self.needs_lazy_update:
            n = self._time_stats["call_count"]

            # Only need to divide if call count is greater than k
            if n>(self.discard_k+1):
                self._time_stats["SpMM"] = self._time_stats["SpMM"] / (n-self.discard_k)
                self._time_stats["GeMM"] = self._time_stats["GeMM"] / (n-self.discard_k)
            self.needs_lazy_update = False
        return self._time_stats

    def forward(self, graph, feat, get_attention=False):
        h_dst = h_src = self.feat_drop(feat)
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # @self.cm.zoomIn(nspace=[self, th])
        # def nb_forward(v):
        #    coeff = [th.exp(self.leaky_relu(nb.el + v.er)) for nb in v.innbs]
        #    s = sum(coeff)
        #    alpha = [c/s for c in coeff]
        #    feat_src = [nb.feat_src for nb in v.innbs]
        #    return sum([alpha[i] * feat_src[i] for i in range(len(feat_src))])
        # rst = nb_forward(g=graph, n_feats= {'el':el, 'er': er, 'feat_src':feat_src})
        rst = B.fused_gat(graph, feat_src, el, er, self.negative_slope)

        # graph = graph.local_var()
        # h_dst = h_src = self.feat_drop(feat)
        # feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # # Vertex-centric implementation.
        # dgl_context = dgl.utils.to_dgl_context(feat.device)
        # graph = graph._graph.get_immutable_gidx(dgl_context)
        # @self.cm.zoomIn(nspace=[self, th])
        # def nb_forward(v):
        #    coeff = [th.exp(self.leaky_relu(nb.el + v.er)) for nb in v.innbs]
        #    s = sum(coeff)
        #    alpha = [c/s for c in coeff]
        #    feat_src = [nb.feat_src for nb in v.innbs]
        #    return sum([alpha[i] * feat_src[i] for i in range(len(feat_src))])
        # rst = nb_forward(g=graph, n_feats= {'el':el, 'er': er, 'feat_src':feat_src})

        rst = rst.squeeze(1)
        # rst = feat_src
        # if self._profile_spmm:
        #     # if device_str == "cuda":
        #     #     th.cuda.synchronize()
        #     t_end = timeit.default_timer()
        #     self.update_timer("SpMM",t_end-t_start)
        if self.activation:
            rst = self.activation(rst)

            # Update call count
        # if self._profile_spmm:
        #     self.update_timer("call_count")
        if get_attention:
            return rst, graph.edata["a"]
        else:
            return rst
    



