from data.text_loader import src_vocab_length, trg_vocab_length
from models.seq2seq import Seq2Seq
from models.transformer import TransformerEncoder, TransformerDecoder
from models.dss import DSSEncoder, DSSDecoder
from models.gated_dss import GatedDSSEncoder, GatedDSSDecoder


def make_transformer(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()

    enc = TransformerEncoder(INPUT_DIM,
                             config.hid_dim,
                             config.enc_layers,
                             config.enc_heads,
                             config.enc_pf_dim,
                             config.enc_dropout,
                             device)

    dec = TransformerDecoder(OUTPUT_DIM,
                             config.hid_dim,
                             config.dec_layers,
                             config.dec_heads,
                             config.dec_pf_dim,
                             config.dec_dropout,
                             device)
    return Seq2Seq(enc, dec, device).to(device)


def make_dss_enc_transformer_dec(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()
    enc = DSSEncoder(INPUT_DIM,
                     config.hid_dim,
                     config.enc_layers,
                     config.enc_dropout,
                     device)
    dec = TransformerDecoder(OUTPUT_DIM,
                             config.hid_dim,
                             config.dec_layers,
                             config.dec_heads,
                             config.dec_pf_dim,
                             config.dec_dropout,
                             device)
    return Seq2Seq(enc, dec, device).to(device)


def make_dss(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()

    enc = DSSEncoder(INPUT_DIM,
                     config.hid_dim,
                     config.enc_layers,
                     config.enc_dropout,
                     device)
    dec = DSSDecoder(OUTPUT_DIM,
                     config.hid_dim,
                     config.dec_layers,
                     config.dec_dropout,
                     device)
    return Seq2Seq(enc, dec, device, decode_mask_type="sequence").to(device)


def make_gated_dss(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()

    enc = GatedDSSEncoder(INPUT_DIM,
                          config.hid_dim,
                          config.enc_layers,
                          config.enc_dropout,
                          device)
    dec = GatedDSSDecoder(OUTPUT_DIM,
                          config.hid_dim,
                          config.dec_layers,
                          config.dec_dropout,
                          device)
    return Seq2Seq(enc, dec, device, decode_mask_type="sequence").to(device)