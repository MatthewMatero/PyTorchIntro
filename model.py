import torch
import torch.nn as nn


class my_lstm(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(my_lstm, self).__init__()

        self.hs = hidden_size
        self.inp = input_size

        self.lstm_cell = nn.LSTMCell(self.inp, self.hs)
        self.act = nn.Softmax(dim=0)
        self.decoder = nn.Linear(self.hs, 4)

        self.run_cuda = torch.cuda.is_available()

    def forward(self, inputs, hidden):
        seq_len = inputs.shape[0] # (len, batch, features)
        hidden_state, cell_state = hidden
        
        hidden_states = []
        for i in range(seq_len):
            hidden_state, cell_state = self.lstm_cell(inputs[i], (hidden_state, cell_state))
            hidden_states.append(hidden_state)
        
        return self.act(self.decoder(hidden_states[-1]))

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden, cell = (weight.new_zeros(batch_size, self.hs), weight.new_zeros(batch_size, self.hs))

        if self.run_cuda:
            hidden.cuda()
            cell.cuda()

        return hidden, cell
