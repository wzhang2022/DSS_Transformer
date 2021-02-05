import spacy
from torchtext.datasets import Multi30k, WMT14
from torchtext.data import Field, BucketIterator

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]



SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)


def get_data_split(dataset):
    if dataset == "Multi30k":
        return Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    elif dataset == "WMT14":
        return WMT14.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    else:
        raise Exception("Invalid data set")


def get_data_iterator_splits(config, device):
    train_data, valid_data, test_data = get_data_split(config.dataset)
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    return BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=config.batch_size,
                device=device)


def get_pad_tokens_idx():
    return SRC.vocab.stoi[SRC.pad_token], TRG.vocab.stoi[TRG.pad_token]


def src_vocab_length():
    return len(SRC.vocab)


def trg_vocab_length():
    return len(TRG.vocab)