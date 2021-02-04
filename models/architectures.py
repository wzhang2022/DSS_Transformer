from data.text_loader import src_vocab_length, trg_vocab_length
from models.seq2seq import Seq2Seq
from models.transformer import TransformerEncoder, TransformerDecoder
from models.dss import DSSEncoder, DSSDecoder


def make_transformer(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = TransformerEncoder(INPUT_DIM,
                             HID_DIM,
                             ENC_LAYERS,
                             ENC_HEADS,
                             ENC_PF_DIM,
                             ENC_DROPOUT,
                             device)

    dec = TransformerDecoder(OUTPUT_DIM,
                             HID_DIM,
                             DEC_LAYERS,
                             DEC_HEADS,
                             DEC_PF_DIM,
                             DEC_DROPOUT,
                             device)
    return Seq2Seq(enc, dec, device).to(device)


def make_dss_enc_transformer_dec(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()
    HID_DIM = 256
    ENC_LAYERS = 5
    DEC_LAYERS = 3
    # ENC_HEADS = 8
    DEC_HEADS = 8
    # ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    enc = DSSEncoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT, device)
    dec = TransformerDecoder(OUTPUT_DIM,
                             HID_DIM,
                             DEC_LAYERS,
                             DEC_HEADS,
                             DEC_PF_DIM,
                             DEC_DROPOUT,
                             device)
    return Seq2Seq(enc, dec, device).to(device)


def make_dss(config, device):
    INPUT_DIM = src_vocab_length()
    OUTPUT_DIM = trg_vocab_length()
    HID_DIM = 256
    ENC_LAYERS = 4
    DEC_LAYERS = 4
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = DSSEncoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT, device)
    dec = DSSDecoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_DROPOUT, device)
    return Seq2Seq(enc, dec, device, decode_mask_type="sequence").to(device)