import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # architectural details
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=3)
    parser.add_argument("--enc_heads", type=int, default=8)
    parser.add_argument("--dec_heads", type=int, default=8)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--enc_pf_dim", type=int, default=512)
    parser.add_argument("--dec_pf_dim", type=int, default=512)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_dropout", type=float, default=0.1)

    # training details
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--clip", type=float, default=1)

    # data details
    parser.add_argument("--dataset", type=str, default="Multi30k")
    return parser.parse_args()
