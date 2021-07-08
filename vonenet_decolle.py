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
target_shape = 10

burnin = 60
starting_epoch = 0
seq_len = 300
device = torch.device("cuda")

vonenet_arch = None
decolle_input_shape = [512, 32, 32]
decolle_arch = [("conv", 64, 3), ("conv", 128, 3), ("conv", 128, 3)]

# vonenet_arch = 'cornets'
# decolle_input_shape = [512]
# decolle_arch = [("dense", 64, 1), ("dense", 128, 1), ("dense", 128, 1)]



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


class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'R', 'U', 'S'])

    def __init__(self, layer_type, in_shape, out_shape, kernel, alpha=.9, beta=.85):
        super(LIFLayer, self).__init__()
        if layer_type == "conv":
            self.base_layer = nn.Conv2d(in_shape[0], out_shape[0], kernel)
        if layer_type == "dense":
            self.base_layer = nn.Linear(in_shape[0], out_shape[0])
        self.in_shape = in_shape
        self.out_shape = out_shape
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
        self.state = self.NeuronState(P=torch.zeros([batch_size] + self.in_shape),
                                      R=torch.zeros([batch_size] + self.out_shape),
                                      U=torch.zeros([batch_size] + self.out_shape),
                                      S=torch.zeros([batch_size] + self.out_shape))

    def forward(self, Sin_t):
        self.cuda()
        state = self.state
        P = self.alpha * state.P + Sin_t
        R = self.alpha * state.R + self.alpha * state.U * state.S
        U = self.base_layer(P) - R
        S = smooth_step(U)
        # The detach function below avoids the backpropagation of error feedback
        self.state = self.NeuronState(P=P.detach(), R=R.detach(), U=U.detach(), S=S.detach())
        return self.state, U, S


def get_out_shape(dim, kernel):
    return int((dim - (kernel - 1) - 1) + 1)

class DECOLLE(nn.Module):

    def __init__(self):
        super().__init__()

        self.LIFs = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.device = device

        in_shape = decolle_input_shape
        for layer_type, n_neurons, kernel in decolle_arch:
            if layer_type == "conv":
                out_shape = [n_neurons, get_out_shape(in_shape[1], kernel), get_out_shape(in_shape[2], kernel)]
            if layer_type == "dense":
                out_shape = [n_neurons]
            self.LIFs.append(LIFLayer(layer_type, in_shape, out_shape, kernel).to(self.device))
            readout = nn.Linear(np.prod(out_shape), target_shape).to(self.device)
            for param in readout.parameters():
                param.requires_grad = False
            self.readouts.append(readout)
            in_shape = out_shape

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

    vonenet = get_model(model_arch=vonenet_arch, pretrained=False, stride=1, sf_max=6, visual_degrees=2).to(device)
    decolle_net = DECOLLE()

    loss_fn = torch.nn.SmoothL1Loss()
    opt = torch.optim.Adam(chain(*[lif.parameters() for lif in decolle_net.LIFs]), lr=1e-5,
                           betas=[0., .95])

    for e in range(starting_epoch, epochs):

        # Main training routine
        for data_batch, target_batch in tqdm.tqdm(iter(gen_train), desc='Epoch {}'.format(e)):
            if target_batch.shape[0] != batch_size:
                break
            data_batch = torch.Tensor(data_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)
            T = data_batch.shape[1]

            decolle_net.reset()
            for k in (range(burnin, T)):
                Sin = data_batch[:, k, :, :].to(device)
                Sin = vonenet(Sin)
                if vonenet_arch == "cornets":
                    Sin = Sin.view(Sin.shape[0], -1)

                for lif, readout in zip(decolle_net.LIFs, decolle_net.readouts):
                    state, u, s = lif.forward(Sin)
                    r = readout(s.view(s.size(0), -1))
                    Sin = s.detach()
                    if k > burnin:
                        loss_t = loss_fn(r, target_batch[:, k, :].to(device))
                        loss_t.backward()
                        opt.step()
                        opt.zero_grad()

        # Test
        predicts = []
        for l in range(len(decolle_arch)):
            predicts.append(np.zeros([seq_len - burnin, batch_size, target_shape]))
        true_pos = [[] for n in  range(len(decolle_arch))]
        for b, (data_batch, target_batch) in enumerate(tqdm.tqdm(iter(gen_test), desc='Testing')):
            if target_batch.shape[0] != batch_size:
                break
            data_batch = torch.Tensor(data_batch).to(device)
            target_batch = torch.Tensor(target_batch).to(device)
            T = data_batch.shape[1]

            decolle_net.reset()
            for k in (range(0, T)):
                target = target_batch[:, k, :]
                Sin = data_batch[:, k, :, :].to(device)
                Sin = vonenet(Sin)
                if vonenet_arch == "cornets":
                    Sin = Sin.view(Sin.size(0), -1)

                for l, (lif, readout) in enumerate(zip(decolle_net.LIFs, decolle_net.readouts)):
                    state, u, s = lif.forward(Sin)
                    r = readout(s.view(s.size(0), -1))
                    Sin = s.detach()
                    if k > burnin:
                        predicts[l][k - burnin] = r.clone().data.cpu().numpy()

            # As there is one readout per timestep, we need to sum the prediction from individual timestep
            for l in range(len(decolle_arch)):
                tp = (predicts[l].sum(0).argmax(-1) == target.cpu().numpy().argmax(-1)).astype(float).tolist()
                true_pos[l] += tp
        test_acc = np.array(true_pos).mean(-1)
        print("Test accuracy: %r" % [acc for acc in test_acc])


if __name__ == "__main__":
    main()
