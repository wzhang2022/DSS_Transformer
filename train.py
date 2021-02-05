import torch
import torch.nn as nn
import numpy as np
import wandb

import random
import math
import time

from data.text_loader import get_data_iterator_splits, get_pad_tokens_idx
from models.architectures import make_transformer, make_dss_enc_transformer_dec, make_dss, make_gated_dss
from utils import parse_args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# set random seeds
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True




def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1], src_pad_idx, trg_pad_idx)
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, _ = get_data_iterator_splits(args, device)
    src_pad_idx, trg_pad_idx = get_pad_tokens_idx()
    with wandb.init(project="DSS_Transformer_UROP", config=args):
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
        print(f'The model has {count_parameters(model):,} trainable parameters')

        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, reduction="mean")
        wandb.watch(model, criterion, log="all", log_freq=1000)
        best_valid_loss = float('inf')

        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, args.clip)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"{args.save_file}.pt")

            wandb.log({"validation_loss": valid_loss})
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
