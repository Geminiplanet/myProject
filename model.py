import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, para, bias=True):
        super(Predictor, self).__init__()
        self.Nseq = para['Nseq']
        self.Nfea = para['Nfea']

        self.hidden_dim = para['hidden_dim']
        self.NLSTM_layer = para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)
        self.encoder_rnn = nn.LSTM(input_size=self.Nfea, hidden_size=self.hidden_dim,
                                   num_layers=self.NLSTM_layer, bias=True,
                                   batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_dim, 24)
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.normal_(self.fc.bias.data)

        for param in self.encoder_rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, X0, L0):
        batch_size = X0.shape[0]
        device = X0.device
        enc_h0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)
        enc_c0 = torch.zeros(self.NLSTM_layer * 1, batch_size, self.hidden_dim).to(device)

        X = self.embedd(X0)
        out, (encoder_hn, encoder_cn) = self.encoder_rnn(X, (enc_h0, enc_c0))
        last_step_index_list = (L0 - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        Z = out.gather(1, last_step_index_list).squeeze()
        # Z=torch.sigmoid(Z)
        Z = F.normalize(Z, p=2, dim=1)

        y = self.fc(Z)  # Z encoder之后的embedding
        return y

    # def Loss(self, y_real, y_pred):
    #     x = y_real - y_pred
    #     x = x ** 2
    #     return torch.sum(x) / y_real.shape[0]
