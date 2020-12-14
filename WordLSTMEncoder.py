import logging

import torch
import torch.nn as nn
from config import config
con = config()

word2vec_path = 'word2vec.txt'


class WordLSTMEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordLSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(con.dropout)
        self.word_dims = 100

        self.word_embed = nn.Embedding(
            vocab.word_size, self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." %
                     (extword_size, word_dims))

        self.extword_embed = nn.Embedding(
            extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.word_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=con.word_hidden_size,
            num_layers=con.word_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, word_ids, extword_ids, batch_masks):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks   sen_num x sent_len

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        # sen_num x sent_len x  hidden*2
        hiddens, _ = self.word_lstm(batch_embed)
        hiddens = hiddens * batch_masks.unsqueeze(2)

        if self.training:
            hiddens = self.dropout(hiddens)

        return hiddens
