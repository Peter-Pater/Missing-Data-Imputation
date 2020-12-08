import torch
import torch.nn as nn

class Mida(nn.Module):
    def __init__(self, n, theta=7, drop_out_ratio=0.5):
        super().__init__()
        self.n = n # dimension of data
        self.theta = theta
        self.drop_out_layer = nn.Dropout(p=drop_out_ratio) # dropout rate is 0.5 as in the paper

        # encoder
        self.encoder_hidden_layer1 = nn.Linear(self.n, self.n + self.theta * 1)
        self.encoder_hidden_layer2 = nn.Linear(self.n + self.theta * 1, self.n + self.theta * 2)
        self.encoder_output_layer = nn.Linear(self.n + self.theta * 2, self.n + self.theta * 3)

        # decoder
        self.decoder_hidden_layer1 = nn.Linear(self.n + self.theta * 3, self.n + self.theta * 2)
        self.decoder_hidden_layer2 = nn.Linear(self.n + self.theta * 2, self.n + self.theta * 1)
        self.decoder_output_layer = nn.Linear(self.n + self.theta * 1, self.n)

    def forward(self, data):
        data = data.view(-1, self.n)
        missing_data = self.drop_out_layer(data)

        z = torch.tanh(self.encoder_hidden_layer1(missing_data)) #pylint: disable=no-member
        z = torch.tanh(self.encoder_hidden_layer2(z)) #pylint: disable=no-member
        encoder_output = self.encoder_output_layer(z)

        z = torch.tanh(self.decoder_hidden_layer1(encoder_output)) #pylint: disable=no-member
        z = torch.tanh(self.decoder_hidden_layer2(z)) #pylint: disable=no-member
        decoder_output = self.decoder_output_layer(z)
        return decoder_output.view(-1, self.n)