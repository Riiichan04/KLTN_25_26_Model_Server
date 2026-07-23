import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GCNConv, GATv2Conv


class EndToEndAspectModel(nn.Module):
    def __init__(self, hidden_channels=64, num_aspects=34, num_classes=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.hidden_channels = hidden_channels

        # 1. Vector đặc trưng ngữ cảnh
        self.fc_in = nn.Linear(768, hidden_channels)
        self.aspect_query = nn.Parameter(torch.randn(num_aspects, hidden_channels))

        # 2. Ba nhánh GCN (Được bổ sung LayerNorm để bảo vệ đặc trưng)
        self.gcn_syn = GCNConv(hidden_channels, hidden_channels)
        self.norm_syn_gcn = nn.LayerNorm(hidden_channels)

        self.gcn_sem = GCNConv(hidden_channels, hidden_channels)
        self.norm_sem_gcn = nn.LayerNorm(hidden_channels)

        self.gcn_asp = GCNConv(hidden_channels, hidden_channels)
        self.norm_asp_gcn = nn.LayerNorm(hidden_channels)

        # 3. Graph Attention Network (GAT) đặt SAU Fusion
        self.heads = 4
        out_channels = hidden_channels * self.heads
        self.gat = GATv2Conv(hidden_channels, hidden_channels, heads=self.heads, concat=True, dropout=0.2)

        # Nén và chuẩn hóa
        self.proj_gat = nn.Linear(out_channels, hidden_channels)
        self.norm_gat = nn.LayerNorm(hidden_channels)

        # 4. Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, input_ids, attention_mask, edge_index_syn, edge_index_sem, edge_index_asp, batch_idx):
        input_ids = input_ids.view(-1, 128)
        attention_mask = attention_mask.view(-1, 128)
        batch_size = input_ids.size(0)

        # --- [BLOCK 1 & 2]: Sinh Vector đặc trưng ngữ cảnh ---
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        word_nodes = self.fc_in(outputs[0])

        aspect_nodes = self.aspect_query.unsqueeze(0).expand(batch_size, -1, -1)
        x_combined = torch.cat([word_nodes, aspect_nodes], dim=1)
        x_flat = x_combined.view(-1, self.hidden_channels)

        # --- [BLOCK 3]: GCN + RESIDUAL + LAYERNORM ---
        # Việc cộng thêm x_flat giúp GCN không làm mất đi tri thức gốc của PhoBERT
        x_syn = F.relu(self.norm_syn_gcn(self.gcn_syn(x_flat, edge_index_syn) + x_flat))
        x_sem = F.relu(self.norm_sem_gcn(self.gcn_sem(x_flat, edge_index_sem) + x_flat))
        x_asp = F.relu(self.norm_asp_gcn(self.gcn_asp(x_flat, edge_index_asp) + x_flat))

        # --- [BLOCK 4]: Fusion Mechanism ---
        x_fused = x_syn + x_sem + x_asp

        # --- [BLOCK 5]: Graph Attention Network ---
        x_gat, (edges_attn, gat_attn) = self.gat(x_fused, edge_index_asp, return_attention_weights=True)

        # --- [BLOCK 6]: Vector đặc trưng tổng hợp ---
        x_synthetic = F.relu(self.norm_gat(self.proj_gat(x_gat) + x_fused))

        # --- [BLOCK 7]: Đưa qua phân loại ---
        x_out = x_synthetic.view(batch_size, -1, self.hidden_channels)
        aspect_features = x_out[:, 128:, :]
        logits = self.classifier(aspect_features)

        return logits, (edges_attn, gat_attn)