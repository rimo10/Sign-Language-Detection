import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class BaseModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels,
                            heads=4, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_channels*4, out_channels, heads=4,
                            concat=False, dropout=0.1)

        self.linear1 = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index, batch):
        out = self.gat1(x, edge_index)
        out = F.relu(out)
        out = self.gat2(out, edge_index)
        # out = global_mean_pool(out, batch)
        out = global_max_pool(out, batch)
        out = self.linear1(out)
        return out
