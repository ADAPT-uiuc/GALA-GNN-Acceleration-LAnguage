"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
import math

import dgl
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from seastar import CtxManager

# pylint: disable=W0235
class TimedSAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            norm="mean",
            weight=True,
            bias=True,
            activation=None,
            allow_zero_in_degree=False,
            use_sddmm=False,
            use_default=True
    ):
        super(TimedSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.use_sddmm = use_sddmm
        self.use_default = use_default

        if weight:
            self.weight_n = nn.Parameter(th.Tensor(in_feats, out_feats))
            self.weight_s = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.activation = activation
        # Seastar
        self.cm = CtxManager(dgl.backend.run_egl)

        self.register_buffer("eps", th.FloatTensor([0.1]))
        

    def reset_parameters(self):
        init.xavier_uniform_(self.weight_s)
        init.xavier_uniform_(self.weight_n)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, norm_data, weight=None, edge_weight=None):
        @self.cm.zoomIn(nspace=[self, th])
        def nb_compute(v):
            feat = sum([nb.h for nb in v.innbs])
            feat = feat * v.norm
            return feat
        feat_n = th.mm(feat, self.weight_n)
        feat_n = nb_compute(g=graph, n_feats={'h' : feat_n, 'norm' : norm_data})
        feat = th.mm(feat, self.weight_s) + feat_n
        # bias
        # if self.bias is not None:
        #     feat = feat + self.bias
        if self.activation:
            feat = self.activation(feat)
        return feat

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)
