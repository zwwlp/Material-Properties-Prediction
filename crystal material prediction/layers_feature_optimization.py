#   Copyright 2019 Takenori Yamamoto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Layer and activation modules for CGNN."""

import torch
import torch.nn as nn
from torch.nn import ( Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU,
                       BatchNorm1d, ModuleList, Sequential,Parameter )
from torch.nn.modules.module import Module
import torch.nn.functional as F

import re
from aggregators import AGGREGATORS
from scalers import SCALERS
from torch import Tensor
from typing import Optional, List, Dict

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    # elif act_name == 'celu':
    #     return CELU(alpha)
    else:
        raise NameError("Not supported activation: {}".format(name))

class SSP(Module):
    r"""Applies element-wise :math:`\text{SSP}(x)=\text{Softplus}(x)-\text{Softplus}(0)`

    Shifted SoftPlus (SSP)

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class BatchNormBilinear(Module):
    """
    Batch Norm Bilinear layer
    """
    def __init__(self, bilinear):
        super(BatchNormBilinear, self).__init__()
        self.in1_features = bilinear.in1_features
        self.in2_features = bilinear.in2_features
        self.out_features = bilinear.out_features
        self.bn = BatchNorm1d(self.out_features)
        self.bilinear = bilinear

    def forward(self, input1, input2):
        output = self.bn(self.bilinear(input1, input2))
        return output

class NodeEmbedding(Module):
    """
    Node Embedding layer
    """
    def __init__(self, in_features, out_features):
        super(NodeEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features, bias=False)

    def forward(self, input):
        output = self.linear(input)
        return output

class EdgeNetwork(Module):
    """
    Edge Network layer
    """
    def __init__(self, in_features, out_features, n_layers, activation=ELU(),
                 use_batch_norm=False, bias=False, use_shortcut=False):
        super(EdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinears = [Bilinear(in_features, in_features, out_features,
                          bias=not use_batch_norm and bias)]
        self.bilinears += [Bilinear(in_features, out_features, out_features,
                           bias=not use_batch_norm and bias)
                           for _ in range(n_layers-1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False)
                               for _ in range(n_layers-1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation

    def forward(self, input, edge_sources, e):
        h = input[edge_sources]
        h = h.contiguous()
        z = e.contiguous()
        for bilinear, shortcut in zip(self.bilinears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = self.activation(bilinear(h, z))
            if shortcut is not None:
                z = z + sc
        return z

def _add_bn(layer):
    return Sequential(layer, BatchNorm1d(layer.out_features))

class FastEdgeNetwork(Module):
    """
    Fast Edge Network layer
    """
    def __init__(self, in_features, out_features, n_layers, activation=ELU(),
                 net_type=0, use_batch_norm=False, bias=False,
                 use_shortcut=False):
        super(FastEdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_linears = [Linear(in_features, out_features,
                             bias=not use_batch_norm and bias)
                             for _ in range(n_layers)]
        self.edge_linears = [Linear(in_features, out_features,
                             bias=not use_batch_norm and bias)]
        self.edge_linears += [Linear(out_features, out_features,
                              bias=not use_batch_norm and bias)
                              for _ in range(n_layers-1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False)
                               for _ in range(n_layers-1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.node_linears = [_add_bn(layer) for layer in self.node_linears]
            self.edge_linears = [_add_bn(layer) for layer in self.edge_linears]
        self.node_linears = ModuleList(self.node_linears)
        self.edge_linears = ModuleList(self.edge_linears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation
        self.net_type = net_type

    def forward(self, input, edge_sources, e):
        z = e
        for node_linear, edge_linear, shortcut in zip(self.node_linears,
            self.edge_linears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = edge_linear(z)
            if self.net_type == 0:
                h = node_linear(input.clone())[edge_sources]
                z = self.activation(h * z)
            else:
                h = self.activation(node_linear(input.clone()))[edge_sources]
                z = h * self.activation(z)
            if shortcut is not None:
                z += sc
        return z

class AggregatedBilinear(Module):
    """
    Aggregated Bilinear layer
    """
    def __init__(self, in1_features, in2_features, out_features,
                 cardinality=32, width=4,
                 activation=ELU(), use_batch_norm=False, bias=False):
        super(AggregatedBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.cardinality = cardinality
        self.width = width
        self.interal_features = cardinality * width
        self.fc_in1 = Linear(in1_features, self.interal_features,
                             bias=not use_batch_norm and bias)
        self.fc_in2 = Linear(in2_features, self.interal_features,
                             bias=not use_batch_norm and bias)
        self.fc_out = Linear(self.interal_features, out_features,
                             bias=not use_batch_norm and bias)
        self.bilinears = [Bilinear(width, width, width,
                          bias=not use_batch_norm and bias)
                          for _ in range(cardinality)]
        if use_batch_norm:
            self.fc_in1 = _add_bn(self.fc_in1)
            self.fc_in2 = _add_bn(self.fc_in2)
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.activation = activation

    def forward(self, input1, input2):
        x1 = self.activation(self.fc_in1(input1))
        x2 = self.activation(self.fc_in2(input2))
        x1 = torch.chunk(x1, self.cardinality, dim=1)
        x2 = torch.chunk(x2, self.cardinality, dim=1)
        output = [bl(c1, c2) for bl, c1, c2 in zip(self.bilinears, x1, x2)]
        output = self.activation(torch.cat(output, dim=1))
        output = self.fc_out(output)
        return output

class AggregatedEdgeNetwork(Module):
    """
    Aggregated Edge Network layer
    """
    def __init__(self, in_features, out_features, n_layers,
                 cardinality=32, width=4, activation=ELU(),
                 use_batch_norm=False, bias=False, use_shortcut=False):
        super(AggregatedEdgeNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinears = [AggregatedBilinear(in_features, in_features, out_features,
                          cardinality=cardinality, width=width,
                          bias=not use_batch_norm and bias)]
        self.bilinears += [AggregatedBilinear(in_features, out_features, out_features,
                           cardinality=cardinality, width=width,
                           bias=not use_batch_norm and bias)
                           for _ in range(n_layers-1)]
        if use_shortcut:
            self.shortcuts = [Linear(in_features, out_features, bias=False)]
            self.shortcuts += [Linear(out_features, out_features, bias=False)
                               for _ in range(n_layers-1)]
        else:
            self.shortcuts = [None for _ in range(n_layers)]
        if use_batch_norm:
            self.bilinears = [BatchNormBilinear(layer) for layer in self.bilinears]
        self.bilinears = ModuleList(self.bilinears)
        self.shortcuts = ModuleList(self.shortcuts)
        self.activation = activation

    def forward(self, input, edge_sources, e):
        h = input[edge_sources]
        z = e
        for bilinear, shortcut in zip(self.bilinears, self.shortcuts):
            if shortcut is not None:
                sc = shortcut(z)
            z = self.activation(bilinear(h, z))
            if shortcut is not None:
                z = z + sc
        return z

class PostconvolutionNetwork(Module):
    """
    Postconvolution Network layer
    """
    def __init__(self, in_features, out_features, n_layers, activation=ELU(),
                 use_batch_norm=False, bias=False):
        super(PostconvolutionNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linears = [Linear(in_features, out_features,
                        bias=not use_batch_norm and bias)]
        self.linears += [Linear(out_features, out_features,
                         bias=not use_batch_norm and bias)
                         for _ in range(n_layers-1)]
        if use_batch_norm:
            self.linears = [_add_bn(layer) for layer in self.linears]
        self.linears = ModuleList(self.linears)
        self.activation = activation

    def forward(self, x):
        for linear in self.linears:
            x = self.activation(linear(x))
        return x

def _bn_act(num_features, activation, use_batch_norm=False):
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation

class GatedGraphConvolution(Module):
    """
    Gated Graph Convolution layer
    """
    def __init__(self, in_features, out_features,n_nbr_fea,dropout, feature_encoding,alpha=0.2,gate_activation=Sigmoid(),
                 node_activation=None, edge_activation=None, edge_network=None,
                 use_node_batch_norm=False, use_edge_batch_norm=False,
                 bias=False, postconv_network=None, use_distance_nbr=False,conv_type=0,use_attention=False):
        super(GatedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_h = None
        self.use_distance_nbr=use_distance_nbr
        self.use_attention=use_attention
        if use_distance_nbr:
            self.linear_nbr=Linear(n_nbr_fea,out_features,bias=False)
        self.use_attention=use_attention
        if self.use_attention:
            self.GraphAttention=GraphAttention(in_features,dropout,alpha)
        if edge_network is None:
            if use_distance_nbr:
                self.linear = Linear(2*in_features, 2 * out_features,
                                     bias=not use_edge_batch_norm and bias)
            else:
                self.linear = Linear(in_features, 2*out_features,
                                 bias=not use_edge_batch_norm and bias)
        else:
            if conv_type > 0:
                linear_out_features = out_features
                self.linear_h = Linear(in_features, out_features,
                                       bias=not use_edge_batch_norm and bias)
            else:
                linear_out_features = 2*out_features
                if use_distance_nbr:
                    self.linear = Linear(in_features+edge_network.out_features, out_features,
                                         bias=not use_edge_batch_norm and bias)
                    self.linear_e=Linear(in_features*2,linear_out_features,bias=not use_edge_batch_norm and bias)
                    self.linear_l = Linear(in_features+edge_network.out_features, linear_out_features,
                                         bias=not use_edge_batch_norm and bias)
                else:
                    self.linear = Linear(edge_network.out_features, linear_out_features,
                                 bias=not use_edge_batch_norm and bias)
        self.gate_activation = _bn_act(out_features, gate_activation,
                                       use_edge_batch_norm)
        self.node_activation = _bn_act(out_features, node_activation,
                                       use_node_batch_norm)
        self.edge_activation = _bn_act(out_features, edge_activation,
                                       use_edge_batch_norm)
        self.edge_network = edge_network
        self.postconv_network = postconv_network
        aggregators = ['sum','mean','min','min_magnitude', 'max', 'max_magnitude','product','var','std','harmonic_mean','Root_mean_square','Euclidean_norm','log_sum_exp']
        aggregators=[number for number, b in zip(aggregators, feature_encoding) if b == True]
        #aggregators=aggregators*feature_encoding
        #scalers = ['identity', 'amplification', 'attenuation']
        #scalers = scalers[scalers_encoding]
        #scalers = [number for number, b in zip(scalers, scalers_encoding) if b == True]
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        #self.scalers = [SCALERS[scale] for scale in scalers]
        #in_channels = (len(self.aggregators) * len(self.scalers)) * in_features
        in_channels = len(self.aggregators) * in_features
        self.aggregators_linear = Linear(in_channels, in_features,bias=bias)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = torch.tensor(8)
        outs = [scaler(out, deg.to(out), self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def forward(self, input, edge_sources, edge_targets, distance_nbr):
        h = input[edge_targets]

        if self.edge_network is None:
            e = h
            if self.use_distance_nbr:
                d_nbr = self.linear_nbr(distance_nbr)
                e = torch.cat([d_nbr, h], dim=1)
                if self.use_attention:
                    source_fea=input[edge_sources]
                    Attention = self.GraphAttention(source_fea, h)
                    z_attention = torch.mul(h, torch.unsqueeze(Attention, 1))
                    e = torch.cat([d_nbr, h+z_attention], dim=1)
        else:
            e = self.edge_network(input, edge_sources, h)
            if self.use_distance_nbr:
                d_nbr = self.linear_nbr(distance_nbr)
                e= torch.cat([e, d_nbr], dim=1)

                if self.use_attention:
                    e = self.linear(e)
                    source_fea=input[edge_sources]
                    Attention = self.GraphAttention(source_fea, h)
                    z_attention = torch.mul(h, torch.unsqueeze(Attention, 1))
                    e = torch.cat([d_nbr, e+z_attention], dim=1)
        if self.linear_h is None:
            if self.use_distance_nbr:
                if self.use_attention:
                    e = self.linear_e(e)
                else:
                    e=self.linear_l(e)
            else:
                e=self.linear(e)
            g, e = torch.chunk(e, 2, dim=1)
        else:
            g = self.linear_e(e)
            e = self.linear_h(h)
        g = self.gate_activation(g)
        if self.edge_activation is not None:
            e = self.edge_activation(e)
        if self.postconv_network is None:
            output = input.clone()
            output.index_add_(0, edge_sources, g * e)
        else:
            # output = torch.zeros_like(input)
            # output.index_add_(0, edge_sources, g * e)
            output=self.aggregate(g * e, edge_sources, input.shape[0])
            output=self.aggregators_linear(output)
            output = input + self.postconv_network(output)
        if self.node_activation is not None:
            output = self.node_activation(output)
        return output

class LinearPooling(Module):
    """
    Linear Pooling layer
    """
    def __init__(self, num_features):
        super(LinearPooling, self).__init__()
        self.num_features = num_features

    def forward(self, input, graph_indices, node_counts):
        graph_count = node_counts.size(0)

        g = torch.zeros(graph_count, self.num_features).to(input.device)
        g.index_add_(0, graph_indices, input)

        n = torch.unsqueeze(node_counts, 1)
        output = g / n
        return output

class GatedPooling(Module):
    """
    Gated Pooling layer
    """
    def __init__(self, num_features, gate_activation=Sigmoid(), bias=True):
        super(GatedPooling, self).__init__()
        self.num_features = num_features
        self.linear = Linear(num_features, num_features, bias=bias)
        self.gate_activation = gate_activation
        self.pooling = LinearPooling(num_features)

    def forward(self, input, graph_indices, node_counts):
        input = self.gate_activation(self.linear(input.clone())) * input.clone()
        output = self.pooling(input, graph_indices, node_counts)
        return output

class GraphPooling(Module):
    """
    Graph Pooling layer
    """
    def __init__(self, num_features, n_steps, activation=Softplus(),
                 use_batch_norm=False):
        super(GraphPooling, self).__init__()
        self.num_features = num_features
        if n_steps > 1:
            self.linears = [Linear(num_features, num_features, bias=False)
                            for _ in range(n_steps)]
            self.linears = ModuleList(self.linears)
        else:
            self.linears = None
        if use_batch_norm:
            self.activation = Sequential(BatchNorm1d(num_features), activation)
        else:
            self.activation = activation

    def forward(self, input):
        if self.linears is not None:
            input = [linear(m) for m, linear in zip(input, self.linears)]
            input = torch.sum(torch.stack(input), 0)
        else:
            input = input[0]
        output = self.activation(input)
        return output

class FullConnection(Module):
    """
    Full Connection layer
    """
    def __init__(self, in_features, out_features, activation=Softplus(),
                 use_batch_norm=False, bias=True):
        super(FullConnection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if use_batch_norm:
            self.linear = Linear(in_features, out_features, bias=False)
            self.activation = Sequential(BatchNorm1d(out_features), activation)
        else:
            self.linear = Linear(in_features, out_features, bias=bias)
            self.activation = activation

    def forward(self, input):
        output = self.activation(self.linear(input))
        return output

class LinearRegression(Module):
    """
    Linear Regression layer
    """
    def __init__(self, in_features, out_features=1):
        super(LinearRegression, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Linear(in_features, out_features)

    def forward(self, input):
        output = self.linear(input)
        return torch.squeeze(output)

class Extension(Module):
    """
    Extension layer
    """
    def __init__(self, num_features=1):
        super(Extension, self).__init__()
        self.num_features = num_features

    def forward(self, input, node_counts):
        if self.num_features > 1:
            n = torch.unsqueeze(node_counts, 1)
        else:
            n = node_counts

        output = input * n
        return output

class GraphAttention(Module):
    def __init__(self,feature_dim,dropout,alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(2*feature_dim,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,center_feature, nbr_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''

        delta_p_concat_h = torch.cat([center_feature,nbr_feature],dim = -1) # [node,source_feature_dim+target_feature_dim]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=1) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, nbr_feature),dim = 1) # [B, npoint, D]
        return graph_pooling
