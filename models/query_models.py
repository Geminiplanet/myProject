import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Ranker:
    def __init__(self):
        self.a = 1

    def ranker(self, rank, sigmoid):
        '''
        ind = int(rank.shape[0] / 2)
        rank_vor = rank[:ind]
        rank_nach = rank[ind:]

        i = rank_vor - rank_nach
        i = i.cpu()
        i_vor = np.int64(i > 0)
        i_nach = np.int64(i < 0)
        ℹ = np.concatenate((i_vor, i_nach), axis=0)
        i = torch.from_numpy(i).float().cuda()
        rank_vers = torch.cat((rank_nach,rank_vor),0)
        if sigmoid:
            rank_diff = torch.sigmoid(rank - rank_vers).detach().cuda()
        else:
            rank_diff = torch.sigmoid(rank - rank_vers).detach().cuda()

        bceloss = nn.BCELoss()
        ranker = bceloss(rank_diff, i)
        '''
        #rank.requires_grad = True
        ranker = torch.mean(-torch.log(rank)).detach().cuda()#+ torch.sum(-torch.log(unlabeled_preds)))
        #torch.autograd.set_detect_anomaly(True)
        #ranker.requires_grad = True
        #ranker = torch.sigmoid(ranker).detach()
        #bceloss = nn.BCELoss()
        #ranker = bceloss(rank_diff, i)
        ranker = torch.clamp(ranker,0, 10)
        ranker.requires_grad = True

        return ranker



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1
        self.ksi = 0.1
        self.ranker = Ranker()


        self.main = nn.Sequential(
          nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(self.dim_h * 2),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(self.dim_h * 4),
          nn.ReLU(True),
          nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 0, bias=False),
          nn.BatchNorm2d(self.dim_h * 8),
          nn.ReLU(True),
        )
        #self.fc = nn.Linear(self.dim_h * 8, self.n_z)
        self.fc_mean = nn.Linear(self.dim_h * 8, self.n_z)
        self.fc_logvar = nn.Linear(self.dim_h * 8, self.n_z)

    def forward(self, rank, x):
        x = self.main(x)
        x = x.squeeze()
        mean = self.fc_mean(x)
        rank = self.ranker.ranker(rank, True).cuda()
        mean = mean + self.ksi * rank
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1
        self.ranker = Ranker()


        self.proj = nn.Sequential(
          nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
          nn.ReLU()
        )

        self.main = nn.Sequential(
          nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
          nn.BatchNorm2d(self.dim_h * 4),
          nn.ReLU(True),
          nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
          nn.BatchNorm2d(self.dim_h * 2),
          nn.ReLU(True),
          nn.ConvTranspose2d(self.dim_h * 2, 3, 8, stride=2),
          nn.Sigmoid()
        )


    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_channel = 3
        self.dim_h = 128
        self.n_z = 1
        self.ksi = 0.1
        self.ranker = Ranker()

        self.main = nn.Sequential(
          nn.Linear(self.n_z, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, self.dim_h * 4),
          nn.ReLU(True),
          nn.Linear(self.dim_h * 4, 1),
          nn.Sigmoid()
        )

    def forward(self, rank, x):
        x = self.main(x)
        rank = self.ranker.ranker(rank, False)
        x = x + self.ksi * rank
        return x


class VAE(nn.Module):
  def __init__(self):
    super(VAE,self).__init__()
    self.encoder=Encoder()
    self.decoder=Decoder()

  def forward(self, rank, x):
    bs = x.size()[0]
    z_mean, z_logvar = self.encoder(rank, x)
    std = z_logvar.mul(0.5).exp_()
    from torch.autograd import Variable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epsilon = Variable(torch.randn(bs, 1)).to(device)
    z = z_mean + std * epsilon

    x_recon = self.decoder(z)
    return x_recon, z_mean,z_logvar
    



