import numpy as np
import torch
import torch.nn
from wideresnet import WideResNet

def tc_loss(fx, s):
    mean = fx.mean(dim=0, keepdim=True)
    sdist = torch.cdist(fx, mean)**2
    intra = torch.diagonal(sdist, dim1=1, dim2=2)
    mask = torch.diag(torch.ones(fx.size(1))).unsqueeze(0) * 1e6
    inter = (sdist + mask).min(-1)[0]
    tc = torch.mean(torch.clamp(intra + s - inter, min=0))
    return tc


class TransClassifier():
    def __init__(self, num_trans, args):
        self.n_trans = num_trans
        self.args = args
        self.netWRN = WideResNet(self.args.depth, num_trans, self.args.widen_factor)
        self.optimizer = torch.optim.Adam(self.netWRN.parameters())


    def fit_trans_classifier(self, x_train, trans_labels, x_test, y_test):
        print('Training')
        self.netWRN.train()
        bs = self.args.batch_size
        N, sh, sw, nc = x_train.shape
        n_rots = self.n_trans
        s = self.args.s
        celoss = torch.nn.CrossEntropyLoss()
        # ndf = 256

        for epoch in range(self.args.epochs):
            print(f'Epoch {epoch}')
            rp = np.random.permutation(N//n_rots)
            rp = np.repeat(rp*n_rots, n_rots) + np.tile(np.arange(n_rots), N//n_rots)
            tot_loss = 0

            for i in range(0,N,bs):
                print(f'Batch {i}')
                x = torch.from_numpy(x_train[rp[i:min(i+bs,N)]])
                y = torch.from_numpy(trans_labels[rp[i:min(i+bs,N)]])
                fx, y_p = self.netWRN(x)
                fx_tc = fx.reshape((bs//n_rots, n_rots, -1))

                tc = tc_loss(fx_tc, s)
                ce = celoss(y_p, y)
                if self.args.reg:
                        loss = ce + self.args.lmbda * tc + 10 *(fx*fx).mean()
                else:
                    loss = ce + self.args.lmbda * tc
                tot_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Loss - {tot_loss*bs/N}')