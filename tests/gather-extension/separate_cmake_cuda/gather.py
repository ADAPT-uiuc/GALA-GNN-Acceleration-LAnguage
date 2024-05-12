import math
import torch

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

# Our module!
# import gather_cpp

torch.manual_seed(42)

torch.ops.load_library("build/libgala_cuda.so")


# print(torch.ops.gala_ops.gather_forward)


class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input_dense,
                offset_graph,
                cols_graph,
                vals_graph,
                weights,
                bias):
        output_res = torch.ops.gala_ops.gather_forward(input_dense,
                                                       offset_graph,
                                                       cols_graph,
                                                       vals_graph,
                                                       weights,
                                                       bias)

        # TODO Normally you pass the intermediate outputs as well.
        #  but not here since there is no "intermediate" outputs.
        #  just pass in the weights for now.
        # TODO bias is not passed here since its pretty much the difference
        #  from the result and input (threshold)
        # variables = output_res + [weights]
        variables = [offset_graph, cols_graph, vals_graph, weights]

        # TODO Only keep this if you are training. Else don't save it.
        ctx.save_for_backward(*variables)
        return output_res

    @staticmethod
    def backward(ctx, grad_h):  # Output of the forward function
        # TODO Only get the first element
        # outputs = gather_cpp.backward(
        #     grad_h.contiguous(), *ctx.saved_tensors)
        # d_input_dense, d_offset_graph, d_cols_graph, d_vals_graph, d_weights, d_bias = outputs
        # d_input_dense = torch.zeros()
        # return d_input_dense, d_offset_graph, d_cols_graph, d_vals_graph, d_weights, d_bias  # Inputs to the forward function
        return outputs[0], torch.zeros(1000000), torch.zeros(1000000), torch.zeros((32, 128)), torch.zeros(128)


class GCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weights = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input_dense, offset_graph, cols_graph, vals_graph):
        return GatherFunction.apply(input_dense,
                                    offset_graph,
                                    cols_graph,
                                    vals_graph,
                                    self.weights,
                                    self.bias)


# class GatherFunction_Tile(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,
#                 nrows,
#                 nvals,
#                 input_dense,
#                 tile_offset_graph,
#                 offset_graph,
#                 rows_graph,
#                 cols_graph,
#                 vals_graph,
#                 weights,
#                 bias):
#         output_res = torch.ops.gala_ops.gather_forward_tile(nrows,
#                                                             nvals,
#                                                             input_dense,
#                                                             tile_offset_graph,
#                                                             offset_graph,
#                                                             rows_graph,
#                                                             cols_graph,
#                                                             vals_graph,
#                                                             weights,
#                                                             bias)
#
#         # TODO Normally you pass the intermediate outputs as well.
#         #  but not here since there is no "intermediate" outputs.
#         #  just pass in the weights for now.
#         # TODO bias is not passed here since its pretty much the difference
#         #  from the result and input (threshold)
#         # variables = output_res + [weights]
#         variables = [offset_graph, cols_graph, vals_graph, weights]
#
#         # TODO Only keep this if you are training. Else don't save it.
#         ctx.save_for_backward(*variables)
#         return output_res
#
#     @staticmethod
#     def backward(ctx, grad_h):  # Output of the forward function
#         # TODO Only get the first element
#         # outputs = gather_cpp.backward(
#         #     grad_h.contiguous(), *ctx.saved_tensors)
#         # d_input_dense, d_offset_graph, d_cols_graph, d_vals_graph, d_weights, d_bias = outputs
#         # d_input_dense = torch.zeros()
#         # return d_input_dense, d_offset_graph, d_cols_graph, d_vals_graph, d_weights, d_bias  # Inputs to the forward function
#         return outputs[0], torch.zeros(1000000), torch.zeros(1000000), torch.zeros((32, 128)), torch.zeros(128)
#
#
# class GCN_Tile(torch.nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCN_Tile, self).__init__()
#         self._in_feats = in_feats
#         self._out_feats = out_feats
#
#         self.weights = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
#         self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weights)
#         torch.nn.init.zeros_(self.bias)
#
#     def forward(self, nrows, nvals, input_dense, tile_offset_graph, offset_graph, rows_graph, cols_graph, vals_graph):
#         return GatherFunction_Tile.apply(nrows, nvals, input_dense,
#                                          tile_offset_graph,
#                                          offset_graph,
#                                          rows_graph,
#                                          cols_graph,
#                                          vals_graph,
#                                          self.weights,
#                                          self.bias)


class GCN_DGL(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN_DGL, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weights = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, graph, feat):
        with graph.local_scope():
            # aggregate_fn = fn.u_mul_e("h0", "dd", "m")
            # graph.srcdata["h0"] = feat
            # graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            # rst = graph.dstdata["h"]

            aggregate_fn = fn.copy_u("h", "m")
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # degs = graph.out_degrees().to(feat_src).clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            # shp = norm.shape + (1,) * (feat_src.dim() - 1)
            # norm = torch.reshape(norm, shp)
            # feat_src = feat_src * norm
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]
            # degs = graph.in_degrees().to(feat_dst).clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            # shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            # norm = torch.reshape(norm, shp)
            # rst = rst * norm

            return rst
