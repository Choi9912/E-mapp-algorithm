# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from torch.cuda import device


class rnnBase(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(rnnBase, self).__init__()
        self.device = device
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, targets):
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])
        outputs = []

        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

            projection = self.project(decoder_output)
            outputs.append(projection)

            decoder_input = torch.LongTensor([targets[i]])

        outputs = torch.stack(outputs).squeeze()
        return outputs

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()

    def print_param(self):

        print('CNN Model Parameters : \n', list(self.model.parameters()))

    def get_model(self):


        return self.__class__

    def close(self):

        del self.__class__
