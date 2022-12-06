import numpy as np
import copy
import torch
import torch.nn
import torch.nn.functional as F
from wideresnet import WideResNet
from sklearn.metrics import roc_auc_score

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

def tc_loss(fx, s):
    mean = fx.mean(dim=0, keepdim=True)
    dists = torch.cdist(fx, mean)**2
    intra = torch.diagonal(dists, dim1=1, dim2=2)
    mask = torch.diag(torch.ones(fx.size(1))).unsqueeze(0).to(device) * 1e6
    inter = (dists + mask).min(-1)[0]
    tc = torch.mean(torch.clamp(intra + s - inter, min=0))
    return tc


class TransClassifier():
    def __init__(self, num_trans, args):
        self.n_trans = num_trans
        self.args = args
        self.netWRN = WideResNet(self.args.depth, num_trans, self.args.widen_factor).to(device)
        self.optimizer = torch.optim.Adam(self.netWRN.parameters())


    def fit_trans_classifier(self, x_train, trans_labels, x_test, y_test):
        print('Training')
        self.netWRN.train()
        bs = self.args.batch_size
        N = x_train.shape[0]
        n_rots = self.n_trans
        s = self.args.s
        celoss = torch.nn.CrossEntropyLoss()
        fxs = []

        best_auc = 0.0
        best_loss = None
        best_model = None
        for epoch in range(self.args.epochs):
            rp = np.random.permutation(N//n_rots)
            rp = np.repeat(rp*n_rots, n_rots) + np.tile(np.arange(n_rots), N//n_rots)
            tot_loss = 0

            for i in range(0,N,bs):
                idx = np.arange(i, min(i+bs,N))
                x = torch.from_numpy(x_train[rp[idx]]).to(device)
                y = torch.from_numpy(trans_labels[rp[idx]]).to(device)
                fx, y_p = self.netWRN(x)
                fx_tc = fx.reshape((bs//n_rots, n_rots, -1))
                fxs.append(fx_tc)

                tc = tc_loss(fx_tc, s)
                ce = celoss(y_p, y)
                if self.args.reg:
                        loss = ce + self.args.lmbda * tc + 10*(fx*fx).mean()
                else:
                    loss = ce + self.args.lmbda * tc
                tot_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = tot_loss*bs/N
            print(f'Epoch {epoch} Loss - {avg_loss}')

            self.netWRN.eval()
            mean = torch.cat(fxs, 0).mean(0)
            N_test = x_test.shape[0]

            with torch.no_grad():
                scores = np.zeros(len(y_test))
                for i in range(0,N_test,bs):
                    idx = range(i,min(i+bs,N_test))
                    x = torch.from_numpy(x_test[idx]).to(device)
                    fx, _ = self.netWRN(x)
                    dists = torch.cdist(fx.reshape((bs,-1)), mean)**2
                    dists = torch.clamp(dists, min=self.args.eps)
                    dists = dists.reshape((bs//n_rots, n_rots, -1))
                    ls_dists = F.log_softmax(-dists, dim=2)

                    reidx = np.arange(bs // n_rots) + i // n_rots
                    scores[reidx] = -torch.diagonal(ls_dists, dim1=1, dim2=2).sum(1).cpu().data.numpy()

                auc = roc_auc_score(y_test, -scores)
                print(f'Epoch {epoch} AUC - {auc}')
                if auc > best_auc:
                    best_auc = auc
                    best_loss = avg_loss
                    best_model = copy.deepcopy(self.netWRN)

        print(f'Best AUC - {best_auc} - @ Loss - {best_loss}')
        if self.args.save_best:
            torch.save(best_model.state_dict(), 'GOAD_trained' + '.pth')

