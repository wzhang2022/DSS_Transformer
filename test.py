import math
import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score
import spacy

from models.architectures import make_transformer, make_dss, make_dss_enc_transformer_dec, make_gated_dss
from data.text_loader import get_data_iterator_splits, get_pad_tokens_idx, get_data_split, SRC, TRG
from utils import parse_args

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1], src_pad_idx, trg_pad_idx)
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor, src_pad_idx)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor, trg_pad_idx, mask_type=model.decode_mask_type)

        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = get_data_iterator_splits(args, device)
    src_pad_idx, trg_pad_idx = get_pad_tokens_idx()

    args = parse_args()
    if args.model == "transformer":
        model = make_transformer(args, device)
    elif args.model == "dss":
        model = make_dss(args, device)
    elif args.model == "dss_enc_transformer_dec":
        model = make_dss_enc_transformer_dec(args, device)
    elif args.model == "gated_dss":
        model = make_gated_dss(args, device)
    else:
        raise Exception("invalid model type")
    model.load_state_dict(torch.load(f"{args.save_file}.pt"))

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    #
    example_idx = 8
    #
    train_data, _, test_data = get_data_split(args.dataset)
    src = vars(train_data.examples[example_idx])['src']
    trg = vars(train_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')




    def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
        trgs = []
        pred_trgs = []

        for datum in data:
            src = vars(datum)['src']
            trg = vars(datum)['trg']

            pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

        return bleu_score(pred_trgs, trgs)


    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f'BLEU score = {bleu_score*100:.2f}')