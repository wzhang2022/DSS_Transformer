import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 device,
                 decode_mask_type="triangular"):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.decode_mask_type = decode_mask_type

    def make_src_mask(self, src, src_pad_idx):
        # src = [batch size, src len]

        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg, trg_pad_idx, mask_type="triangular"):
        # trg = [batch size, trg len]

        if mask_type == "triangular":
            trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
            # trg_pad_mask = [batch size, 1, 1, trg len]

            trg_len = trg.shape[1]
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
            # trg_sub_mask = [trg len, trg len]

            trg_mask = trg_pad_mask & trg_sub_mask
            # trg_mask = [batch size, 1, trg len, trg len]

            return trg_mask
        elif mask_type == "sequence":
            trg_mask = self.make_src_mask(trg, trg_pad_idx)
            # trg_mask = [batch size, 1, 1, trg len]
            return trg_mask
        else:
            raise Exception("Invalid mask_type")

    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src, src_pad_idx)
        trg_mask = self.make_trg_mask(trg, trg_pad_idx, self.decode_mask_type)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, hid dim]

        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]

        return output