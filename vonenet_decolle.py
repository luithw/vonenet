import torch

from vonenet import get_model

import datetime
import os
from itertools import chain

from collections import namedtuple
import torch.nn as nn
import torch
from torchneuromorphic.nmnist import nmnist_dataloaders
import numpy as np
import tqdm


batch_size = 100
epochs = 100
input_shape = 1000
target_shape = 10
arch = [128, 128]
burnin = 60
starting_epoch = 0
seq_len = 300
device = torch.device("cuda")


class SmoothStep(torch.autograd.Function):
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).float()

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input

smooth_step = SmoothStep().apply


class LIFDenseLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'R', 'U', 'S'])

    def __init__(self, in_channels, out_channels, bias=True, alpha=.9, beta=.85):
        super(LIFDenseLayer, self).__init__()
        self.base_layer = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.reset()
        self.base_layer.weight.data.uniform_(-.3, .3)
        self.base_layer.bias.data.uniform_(-.01, .01)

    def cuda(self, device=None):
        self = super().cuda(device)
        self.state = self.NeuronState(P=self.state.P.cuda(device),
                                      R=self.state.R.cuda(device),
                                      U=self.state.U.cuda(device),
                                      S=self.state.S.cuda(device))
        return self

    def reset(self):
        self.state = self.NeuronState(P=torch.zeros(batch_size, self.in_channels),
                                      R=torch.zeros(batch_size, self.out_channels),
                                      U=torch.zeros(batch_size, self.out_channels),
                                      S=torch.zeros(batch_size, self.out_channels))

    def forward(self, Sin_t):
        Sin_t = Sin_t.view(Sin_t.size(0), -1)
        self.cuda()
        state = self.state
        P = self.alpha * state.P + Sin_t
        R = self.alpha * state.R + self.alpha * state.U * state.S
        U = self.base_layer(P) - R
        S = smooth_step(U)
        # The detach function below avoids the backpropagation of error feedback
        self.state = self.NeuronState(P=P.detach(), R=R.detach(), U=U.detach(), S=S.detach())
        return self.state, U, S


class DECOLLE(nn.Module):

    def __init__(self):
        super().__init__()

        self.LIFs = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.device = device

        in_shape = input_shape
        for n_neurons in arch:
            self.LIFs.append(LIFDenseLayer(in_channels=in_shape, out_channels=n_neurons).to(self.device))
            readout = nn.Linear(n_neurons, target_shape).to(self.device)
            for param in readout.parameters():
                param.requires_grad = False
            self.readouts.append(readout)
            in_shape = n_neurons

    def reset(self):
        for lif in self.LIFs:
            lif.reset()

    def __len__(self):
        return len(self.LIFs)


def main():
    gen_train, gen_test = nmnist_dataloaders.create_dataloader(root='data/n_mnist.hdf5',
                                                               chunk_size_train=seq_len,
                                                               chunk_size_test=seq_len,
                                                               batch_size=batch_size,
                                                               dt=1000,
                                                               num_workers=4)

    vonenet = get_model(model_arch='cornets', pretrained=False).to(device)
    decolle = DECOLLE()

    loss_fn = torch.nn.SmoothL1Loss()
    opt = torch.optim.Adam(chain(*[lif.parameters() for lif in decolle.LIFs]), lr=1e-5,
                           betas=[0., .95])

    for e in range(starting_epoch, epochs):

        # Main training routine
        for data_batch, target_batch in tqdm.tqdm(iter(gen_train), desc='Epoch {}'.format(e)):
            if target_batch.shape[0] != batch_size:
                break
            data_batch = torch.Tensor(data_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)
            T = data_batch.shape[1]

            decolle.reset()
            for k in (range(burnin, T)):
                Sin = data_batch[:, k, :, :].to(device)
                Sin = vonenet(Sin).detach()

                for lif, readout in zip(decolle.LIFs, decolle.readouts):
                    state, u, s = lif.forward(Sin)
                    r = readout(s)
                    Sin = s.detach()
                    if k > burnin:
                        loss_t = loss_fn(r, target_batch[:, k, :].to(device))
                        loss_t.backward()
                        opt.step()
                        opt.zero_grad()

        # Test
        predicts = []
        for l in arch:
            predicts.append(np.zeros([seq_len - burnin, batch_size, target_shape]))
        true_pos = [[] for n in arch]
        for b, (data_batch, target_batch) in enumerate(tqdm.tqdm(iter(gen_test), desc='Testing')):
            if target_batch.shape[0] != batch_size:
                break
            data_batch = torch.Tensor(data_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)
            T = data_batch.shape[1]

            decolle.reset()
            for k in (range(0, T)):
                target = target_batch[:, k, :]
                Sin = data_batch[:, k, :, :].to(device)
                Sin = vonenet(Sin).detach()
                for l, (lif, readout) in enumerate(zip(decolle.LIFs, decolle.readouts)):
                    state, u, s = lif.forward(Sin)
                    r = readout(s)
                    Sin = s.detach()
                    if k > burnin:
                        predicts[l][k - burnin] = r.clone().data.cpu().numpy()

            # As there is one readout per timestep, we need to sum the prediction from individual timestep
            for l in range(len(arch)):
                tp = (predicts[l].sum(0).argmax(-1) == target.cpu().numpy().argmax(-1)).astype(float).tolist()
                true_pos[l] += tp
        test_acc = np.array(true_pos).mean(-1)
        print("Test accuracy: %r" % [acc for acc in test_acc])

if __name__ == "__main__":
    main()
