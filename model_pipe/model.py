from torch import nn
from torch_geometric_temporal.nn.attention import ASTGCN


class DART_GN_HCMC(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, heads, hidden_units, dropout, device):
        super(DART_GNN_HCMC, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.device = device




class model_beuteu(nn.Module):
    def __init__(self, nb_block = 2, in_channels = 3, K = 3, nb_chev_filter = 64, nb_time_filter = 64, time_strides = 1, num_for_predict = 1, len_input = 20, num_of_vertices = 24):
        super(model_beuteu, self).__init__()
        self.nb_block = nb_block
        self.in_channels = in_channels
        self.K = K
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.num_for_predict = num_for_predict
        self.len_input = len_input
        self.num_of_vertices = num_of_vertices
        self.ASTCGN = ASTGCN( nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input, num_of_vertices, normalization='sym')

    def forward(self, input_x, edge_index):
        out = self.ASTCGN(input_x, edge_index, edge_weight=None)
        return out
