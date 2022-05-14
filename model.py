import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import GRU
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.utils import softmax


class MultiHeadTripletAttention(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs)  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = nn.Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = nn.Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = nn.Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = nn.Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = nn.Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_node)
        nn.init.kaiming_uniform_(self.weight_edge)
        nn.init.kaiming_uniform_(self.weight_triplet_att)
        nn.init.kaiming_uniform_(self.weight_scale)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class Block(nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, latent_dim, depth=3, heads=4, dropout=0.1, outdim=2):
        super(Encoder, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            Block(hidden_dim, edge_dim, heads)
            for i in range(depth)
        ])
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.encoder_1 = nn.Linear(128, latent_dim)
        self.encoder_2 = nn.Linear(128, latent_dim)

    def forward(self, data):
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + F.dropout(conv(x, data.edge_index, data.edge_attr), p=self.dropout, training=self.training)
        x = self.set2set(x, data.batch)
        z_mu = self.encoder_1(x)
        z_lsgms = self.encoder_2(x)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        # eps = Variable(torch.randn(z_sgm.size())).cuda()
        eps = torch.randn(z_sgm.size())
        z = eps * z_sgm + z_mu  #
        return z

# class Decoder(nn.Module):
#     def __init__(self, conv_dim, z_dim, ):
