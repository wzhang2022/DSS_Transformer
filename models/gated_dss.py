import torch
import torch.nn as nn


class GatedDSSEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.siamese = nn.Linear(hid_dim, hid_dim)
        self.global_layer = nn.Linear(hid_dim, hid_dim)
        self.gating_layer_siamese = nn.Linear(hid_dim, hid_dim)
        self.gating_layer_global = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        _src = self.siamese(src)
        # _src = [batch size, src len, hid dim]

        src_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(2)
        # src_mask = [batch size, src len, 1]

        global_vec = torch.sum(src * src_mask, dim=1) / torch.sum(src_mask, dim=1)
        # global_vec = [batch size, hid dim]

        global_vec = self.global_layer(global_vec).unsqueeze(1)
        # global_vec = [batch size, 1, hid dim]

        # gating
        gate = torch.sigmoid(self.gating_layer_siamese(_src) + self.gating_layer_global(global_vec))
        src = _src * gate + global_vec * (1 - gate)
        # src = [batch size, src len, hid dim]

        # activation, dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(torch.relu(src)))
        # src = [batch size, src len, hid dim]

        return src


class GatedDSSDecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 dropout):
        super().__init__()

        # autoregressive trg net
        self.dec_layer_norm = nn.LayerNorm(hid_dim)
        self.dec_siamese = nn.Linear(hid_dim, hid_dim)
        self.dec_global_layer = nn.Linear(hid_dim, hid_dim)
        self.dec_gate_siamese = nn.Linear(hid_dim, hid_dim)
        self.dec_gate_global = nn.Linear(hid_dim, hid_dim)

        # src net
        self.enc_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_siamese = nn.Linear(hid_dim, hid_dim)
        self.enc_global_layer = nn.Linear(hid_dim, hid_dim)
        self.enc_gate_siamese = nn.Linear(hid_dim, hid_dim)
        self.enc_gate_global = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, 1, trg len]
        # src_mask = [batch size, 1, 1, src len]

        _trg = self.dec_siamese(trg)
        # trg = [batch size, trg len, hid dim]

        trg_mask = trg_mask.squeeze(1).squeeze(1).unsqueeze(2)
        # trg_mask = [batch size, trg len, 1]

        dec_global_vec = (torch.cumsum(trg, dim=1) / torch.cumsum(trg_mask, dim=1)) * trg_mask
        # dec_global_vec = [batch_size, trg len, hid dim]
        # this computes a rolling average and zeros out pad tokens

        dec_global_vec = self.dec_global_layer(dec_global_vec)
        # dec_global_vec = [batch_size, trg len, hid dim]

        # trg gating
        dec_gate = torch.sigmoid(self.dec_gate_siamese(_trg) + self.enc_gate_global(dec_global_vec))
        trg = _trg * dec_gate + dec_global_vec * (1 - dec_gate)
        # trg = [batch size, trg len, hid dim]


        # activation, dropout, residual and layer norm
        trg = self.dec_layer_norm(trg + self.dropout(torch.relu(trg)))
        # trg = [batch size, trg len, hid dim]

        # _enc_src = self.enc_siamese(enc_src)
        # _src = [batch size, src len, hid dim]

        src_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(2)
        # src_mask = [batch size, src len, 1]

        enc_global_vec = torch.sum(enc_src * src_mask, dim=1) / torch.sum(src_mask, dim=1)
        # global_vec = [batch_size, hid dim]

        enc_global_vec = self.enc_global_layer(enc_global_vec).unsqueeze(1)
        # global_vec = [batch_size, 1, hid dim]

        # trg gating
        enc_gate = torch.sigmoid(self.enc_gate_siamese(trg) + self.enc_gate_global(enc_global_vec))
        trg = _trg * enc_gate + enc_global_vec * (1 - enc_gate)
        # trg = [batch size, trg len, hid dim]

        # activation, dropout, residual and layer norm
        trg = self.enc_layer_norm(trg + self.dropout(torch.relu(trg)))
        # trg = [batch size, trg len, hid dim]

        return trg


class GatedDSSEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([GatedDSSEncoderLayer(hid_dim,
                                                     dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class GatedDSSDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([GatedDSSDecoderLayer(hid_dim,
                                                     dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]

        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        return output
