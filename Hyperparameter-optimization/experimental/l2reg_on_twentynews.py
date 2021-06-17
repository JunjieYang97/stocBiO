from itertools import repeat
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized
from stocBiO import *

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import hypergrad as hg
import numpy as np
import time
import os

class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help='epoch numbers')
    parser.add_argument('--T', default=10, type=int, help='inner update iterations')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--hessian_q', type=int, default=10, help='number of steps to approximate hessian')
    # Only when alg == minibatch, we apply stochastic, otherwise, alg training with full batch
    parser.add_argument('--alg', type=str, default='reverse', choices=['stocBiO', 'reverse', 'AID-FP', 'AID-CG'])
    parser.add_argument('--training_size', type=int, default=2000)
    parser.add_argument('--inner_lr', type=float, default=100.0)
    parser.add_argument('--inner_mu', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=100.0)
    parser.add_argument('--outer_mu', type=float, default=0.0)
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    args = parser.parse_args()
    # outer_lr, outer_mu = 100.0, 0.0  # nice with 100.0, 0.0 (torch.SGD) tested with T, K = 5, 10 and CG
    # inner_lr, inner_mu = 100., 0.9   # nice with 100., 0.9 (HeavyBall) tested with T, K = 5, 10 and CG
    # parser.add_argument('--seed', type=int, default=0)

    if args.alg == 'stocBiO':
        args.batch_size = args.batch_size
    else:
        args.batch_size = args.training_size
        args.val_size = args.training_size

    if not args.save_folder:
        args.save_folder = './save_results'
    args.model_name = '{}_bs_{}_vbs_{}_olrmu_{}_{}_ilrmu_{}_{}_eta_{}_T_{}_hessianq_{}'.format(args.alg, 
                       args.batch_size, args.val_size, args.outer_lr, args.outer_mu, args.inner_lr, 
                       args.inner_mu, args.eta, args.T, args.hessian_q)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    # parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    return args


def train_model(args):

    # Constant
    tol = 1e-12
    warm_start = True
    bias = False  # without bias outer_lr can be bigger (much faster convergence)
    train_log_interval = 100
    val_log_interval = 1

    # Basic Setting 
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cuda = True and torch.cuda.is_available()
    default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    torch.set_default_tensor_type(default_tensor_str)
    #torch.multiprocessing.set_start_method('forkserver')

    # Functions 
    def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
    def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()
    def train_loss(params, hparams, data):
        x_mb, y_mb = data
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  params)
        return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)
    def val_loss(opt_params, hparams):
        x_mb, y_mb = next(val_iterator)
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  opt_params[:len(parameters)])
        val_loss = F.cross_entropy(out, y_mb)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

        val_losses.append(tonp(val_loss))
        val_accs.append(acc)
        return val_loss
    def reg_f(params, l2_reg_params, l1_reg_params=None):
        r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
        if l1_reg_params is not None:
            r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
        return r
    def out_f(x, params):
        out = x @ params[0]
        out += params[1] if len(params) == 2 else 0
        return out
    def eval(params, x, y):
        out = out_f(x,  params)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

        return loss, acc

    
    # load twentynews and preprocess
    val_size_ratio = 0.5
    X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True,
                                        #remove=('headers', 'footers', 'quotes')
                                        )
    x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True,
                                                #remove=('headers', 'footers', 'quotes')
                                                )
    x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)
    train_samples, n_features = x_train.shape
    test_samples, n_features = x_test.shape
    val_samples, n_features = x_val.shape
    n_classes = np.unique(y_train).shape[0]
    # train_samples=5657, val_samples=5657, test_samples=7532, n_features=130107, n_classes=20
    print('Dataset 20newsgroup, train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
        % (train_samples, val_samples, test_samples, n_features, n_classes))
    ys = [frnp(y_train).long(), frnp(y_val).long(), frnp(y_test).long()]
    xs = [x_train, x_val, x_test]

    if cuda:
        xs = [from_sparse(x).cuda() for x in xs]
    else:
        xs = [from_sparse(x) for x in xs]

    # x_train.size() = torch.Size([5657, 130107])
    # y_train.size() = torch.Size([5657])
    x_train, x_val, x_test = xs
    y_train, y_val, y_test = ys
    
    # torch.DataLoader has problems with sparse tensor on GPU    
    iterators, train_list, val_list = [], [], []
    xmb_train, xmb_val, ymb_train, ymb_val = [], [], [], []
    # For minibatch method, we build the list to store the splited tensor
    if args.alg == 'stocBiO':
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(CustomTensorIterator([x, y], batch_size=args.batch_size, shuffle=True, **kwargs))
        train_iterator, val_iterator = iterators
        for _ in range(train_samples // args.batch_size+1):
            data_temp = next(train_iterator)
            x_mb, y_mb = data_temp
            xmb_train.append(x_mb)
            ymb_train.append(y_mb)
            # train_list.append(next(train_iterator))
        for _ in range(val_samples // args.val_size+1):
            data_temp = next(val_iterator)
            x_mb, y_mb = data_temp
            xmb_val.append(x_mb)
            ymb_val.append(y_mb)
            # val_list.append(next(val_iterator))
        train_list, val_list = [xmb_train, ymb_train], [xmb_val, ymb_val]
        train_list_len, val_list_len = len(ymb_train), len(ymb_val)

        # set up another train_iterator & val_iterator to make sure train_list and val_list are full
        iterators = []
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(repeat([x, y]))
        train_iterator, val_iterator = iterators
    else:
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(repeat([x, y]))
        train_iterator, val_iterator = iterators

       
    # Initialize parameters
    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
    l1_reg_params = (0.*torch.ones(1)).requires_grad_(True)  # one l1 hp only (best when really low)
    #l2_reg_params = (-20.*torch.ones(1)).requires_grad_(True)  # one l2 hp only (best when really low)
    #l1_reg_params = (-1.*torch.ones(n_features)).requires_grad_(True)
    hparams = [l2_reg_params]
    # hparams: the outer variables (or hyperparameters)
    ones_dxc = torch.ones(n_features, n_classes)

    outer_opt = torch.optim.SGD(lr=args.outer_lr, momentum=args.outer_mu, params=hparams)
    # outer_opt = torch.optim.Adam(lr=0.01, params=hparams)

    params_history = []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    w = torch.zeros(n_features, n_classes).requires_grad_(True)
    parameters = [w]

    # params_history: the inner iterates (from first to last)
    if bias:
        b = torch.zeros(n_classes).requires_grad_(True)
        parameters.append(b)
 
    if args.inner_mu > 0:
        #inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
        inner_opt = hg.HeavyBall(train_loss, args.inner_lr, args.inner_mu, data_or_iter=train_iterator)
    else:
        inner_opt = hg.GradientDescent(train_loss, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(train_loss, 1., data_or_iter=train_iterator)

    total_time = 0
    loss_acc_time_results = np.zeros((args.epochs+1, 3))
    test_loss, test_acc = eval(parameters, x_test, y_test)
    loss_acc_time_results[0, 0] = test_loss
    loss_acc_time_results[0, 1] = test_acc
    loss_acc_time_results[0, 2] = 0.0
    
    for o_step in range(args.epochs):
        start_time = time.time()
        if args.alg == 'stocBiO':
            # train_index_list = torch.randperm(train_list_len)
            # val_index = torch.randperm(val_list_len)
            inner_losses = []
            for t in range(args.T):
                # loss_train = train_loss(parameters, hparams, train_list[train_index_list[t%train_list_len]])
                loss_train = train_loss(parameters, hparams, [xmb_train[t%train_list_len], ymb_train[t%train_list_len]])
                inner_grad = torch.autograd.grad(loss_train, parameters)
                parameters[0] = parameters[0] - args.inner_lr*inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == args.T-1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            outer_update = stocbio(parameters, hparams, val_list, args, out_f, reg_fs)

            hparams[0] = hparams[0] - args.outer_lr*outer_update
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        else:
            inner_losses = []
            if params_history:
                params_history = [params_history[-1]]
            else:
                params_history = [inner_opt.get_opt_params(parameters)]
            for t in range(args.T):
                params_history.append(inner_opt(params_history[-1], hparams, create_graph=False))
                inner_losses.append(inner_opt.curr_loss)

                if t % train_log_interval == 0 or t == args.T-1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            final_params = params_history[-1]
            outer_opt.zero_grad()
            if args.alg == 'reverse':
                hg.reverse(params_history[-args.hessian_q-1:], hparams, [inner_opt]*args.hessian_q, val_loss)
            elif args.alg == 'AID-FP':
                hg.fixed_point(final_params, hparams, args.hessian_q, inner_opt, val_loss, stochastic=False, tol=tol)
            # elif args.alg == 'neuman':
            #     hg.neumann(final_params, hparams, args.K, inner_opt, val_loss, tol=tol)
            elif args.alg == 'AID-CG':
                hg.CG(final_params[:len(parameters)], hparams, args.hessian_q, inner_opt_cg, val_loss, stochastic=False, tol=tol)
            outer_opt.step()

            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)

        iter_time = time.time() - start_time
        total_time += iter_time
        if o_step % val_log_interval == 0 or o_step == args.T-1:
            test_loss, test_acc = eval(final_params[:len(parameters)], x_test, y_test)
            loss_acc_time_results[o_step+1, 0] = test_loss
            loss_acc_time_results[o_step+1, 1] = test_acc
            loss_acc_time_results[o_step+1, 2] = total_time
            print('o_step={} ({:.2e}s) Val loss: {:.4e}, Val Acc: {:.2f}%'.format(o_step, iter_time, val_losses[-1],
                                                                                100*val_accs[-1]))
            print('          Test loss: {:.4e}, Test Acc: {:.2f}%'.format(test_loss, 100*test_acc))
            print('          l2_hp norm: {:.4e}'.format(torch.norm(hparams[0])))
            if len(hparams) == 2:
                print('          l1_hp : ', torch.norm(hparams[1]))

    file_name = 'results.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)   

    print(loss_acc_time_results)
    print('HPO ended in {:.2e} seconds\n'.format(total_time))


def from_sparse(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def train_loss(params, hparams, data):
    x_mb, y_mb = data
    out = out_f(x_mb,  params)
    return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)

def val_loss(opt_params, hparams):
    x_mb, y_mb = next(val_iterator)
    out = out_f(x_mb,  opt_params[:len(parameters)])
    val_loss = F.cross_entropy(out, y_mb)
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

    val_losses.append(tonp(val_loss))
    val_accs.append(acc)
    return val_loss

def reg_f(params, l2_reg_params, l1_reg_params=None):
    ones_dxc = torch.ones(params[0].size())
    r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
    if l1_reg_params is not None:
        r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
    return r

def reg_fs(params, hparams, loss):
    reg = reg_f(params, *hparams)
    return loss+reg

def out_f(x, params):
    out = x @ params[0]
    out += params[1] if len(params) == 2 else 0
    return out


def main():
    args = parse_args()
    print(args)
    train_model(args)

if __name__ == '__main__':
    main()
