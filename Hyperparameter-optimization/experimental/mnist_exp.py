import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import tensorboard_logger as tb_logger
import hypergrad as hg
import time

from itertools import repeat
from torch.nn import functional as F
from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=15, help='K')
    parser.add_argument('--iterations', type=int, default=200, help='T')
    parser.add_argument('--outer_lr', type=float, default=1, help='beta')
    parser.add_argument('--inner_lr', type=float, default=0.1, help='alpha')
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--data_path', default='data/', help='The temporary data storage path')
    parser.add_argument('--training_size', type=int, default=20000)
    parser.add_argument('--validation_size', type=int, default=5000)
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--hessian_q', type=int, default=3)
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='minibatch', choices=['minibatch', 'GD', '2timescale', '1sample', 
                                                                'reverse', 'CG', 'fixed_point'])
    args = parser.parse_args()
    

    if args.alg == 'minibatch':
        args.batch_size = args.batch_size
    elif args.alg == '1sample':
        args.batch_size = 1
    elif args.alg == '2timescale':
        args.batch_size = 1
        args.iterations = 1
    else:
        args.batch_size = args.training_size
    
    if not args.save_folder:
        args.save_folder = './save_tb_results'
    args.model_name = '{}_{}_bs_{}_olr_{}_ilr_{}_eta_{}_noiser_{}_q_{}_ite_{}'.format(args.alg, 
                       args.training_size, args.batch_size, args.outer_lr, args.inner_lr, args.eta, 
                       args.noise_rate, args.hessian_q, args.iterations)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    return args


def get_data_loaders(args):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    train_sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
        batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=args.data_path, train=False,
                        download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=args.test_size)
    return train_loader, test_loader


def train_model(args, train_loader, test_loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger = tb_logger.Logger(logdir=args.save_folder)
    parameters = torch.randn((args.num_classes, 785), requires_grad=True)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out').to(device)
    lambda_x = torch.zeros((args.training_size), requires_grad=True).to(device)
    loss_time_results = np.zeros((args.epochs+1, 3))
    batch_num = args.training_size//args.batch_size

    batch_val_num = args.validation_size//args.batch_size
    # lambda_x = torch.randn((args.training_size), requires_grad=True).to(device)

    train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
    test_loss_avg = loss_test_avg(test_loader, parameters, device)
    logger.log_value('Train_loss_epoch', train_loss_avg, 0)
    logger.log_value('Test_loss_epoch', test_loss_avg, 0)
    logger.log_value('Train_loss_time', train_loss_avg, 0)
    logger.log_value('Test_loss_time', test_loss_avg, 0)
    loss_time_results[0, 0] = train_loss_avg
    loss_time_results[0, 1] = test_loss_avg
    loss_time_results[0, 2] = (0.0)
    print('Epoch: {:d} Train Loss: {:.4f} Test Loss: {:.4f}'.format(0, train_loss_avg, test_loss_avg))

    lambda_index_outer = 0
    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(train_loader):
        images_list.append(images)
        labels_list.append(labels)

    # setting for reverse, fixed_point & CG
    def loss_inner(parameters, weight, data_all):
        data = data_all[0]
        labels = data_all[1]
        data = torch.reshape(data, (data.size()[0],-1)).to(device)
        labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
        output = torch.matmul(data, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        loss = F.cross_entropy(output, labels_cp, reduction='none')
        loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(weight[0]))) + 0.001*torch.pow(torch.norm(parameters[0]),2)
        return loss_regu

    def loss_outer(parameters, lambda_x):
        images, labels = images_list[-1], labels_list[-1]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
        images = torch.cat([images_temp]*(args.training_size // args.validation_size))
        labels = torch.cat([labels_temp]*(args.training_size // args.validation_size))
        output = torch.matmul(images, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        loss = F.cross_entropy(output, labels)
        return loss

    tol = 1e-12
    warm_start = True
    params_history = []
    train_iterator = repeat([images_list[0], labels_list[0]])
    inner_opt = hg.GradientDescent(loss_inner, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(loss_inner, 1., data_or_iter=train_iterator)
    outer_opt = torch.optim.SGD(lr=args.outer_lr, params=[lambda_x])
    
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.alg == 'minibatch' or args.alg == 'GD':
            train_index_list = torch.randperm(batch_num)
            for index in range(args.iterations):
                index_rn = train_index_list[index%batch_num]
                images, labels = images_list[index_rn], labels_list[index_rn]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
                labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
                weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight)
                parameters = parameters - args.inner_lr*inner_update
        
        if args.alg == 'minibatch':
            val_index = -torch.randperm(batch_val_num)
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            
            # Fy_gradient
            images, labels = images_list[val_index[1]], labels_list[val_index[1]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            Fy_gradient = gradient_fy(args, labels, parameters, images)
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            images, labels = images_list[val_index[2]], labels_list[val_index[2]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            G_gradient = torch.reshape(parameters, [-1]) - args.eta*Gy_gradient
            for q in range(args.hessian_q):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)

            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)
            # Gyx_gradient
            images, labels = images_list[val_index[0]], labels_list[val_index[0]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), weight)[0]
            outer_update = -Gyx_gradient
        
        elif args.alg == 'GD':
            images, labels = images_list[-1], labels_list[-1]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
            images = torch.cat([images_temp]*(args.training_size // args.validation_size))
            labels = torch.cat([labels_temp]*(args.training_size // args.validation_size))

            # Fy_gradient
            labels = labels.to(device)
            Fy_gradient = gradient_fy(args, labels, parameters, images)
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            G_gradient = torch.reshape(parameters, [-1]) - args.eta*Gy_gradient
            for q in range(args.hessian_q):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)

            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), weight)[0]
            outer_update = -Gyx_gradient

        elif args.alg == '1sample' or args.alg == '2timescale':
            train_index_list = torch.randperm(args.training_size)
            random_list = np.random.uniform(size=[args.training_size])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)
            for index in range(args.iterations):
                images, labels = images_list[train_index_list[index]], labels_list[train_index_list[index]]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
                labels_cp = nositify(labels, noise_rate_list[index], args.num_classes).to(device)
                weight = lambda_x[train_index_list[index]: train_index_list[index]+1]
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight)
                parameters = parameters - args.inner_lr*inner_update

            val_index = -torch.randperm(args.validation_size)
            random_list = np.random.uniform(size=[args.hessian_q+2])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)

            # Fy_gradient
            images, labels = images_list[val_index[1]], labels_list[val_index[1]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            Fy_gradient = gradient_fy(args, labels, parameters, images)
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            images, labels = images_list[val_index[2]], labels_list[val_index[2]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, noise_rate_list[1], args.num_classes).to(device)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            G_gradient = torch.reshape(parameters, [-1]) - args.eta*Gy_gradient
            for q in range(args.hessian_q):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)            
            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)

            # Gyx_gradient
            images, labels = images_list[val_index[0]], labels_list[val_index[0]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, noise_rate_list[0], args.num_classes).to(device)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), weight)[0]
            outer_update = -Gyx_gradient      
        else:
            inner_losses = []
            if params_history:
                params_history = [params_history[-1]]
            else:
                params_history = [[parameters]]
            for index in range(args.iterations):
                params_history.append(inner_opt(params_history[-1], [lambda_x], create_graph=False))
                inner_losses.append(inner_opt.curr_loss)

            final_params = params_history[-1]
            outer_opt.zero_grad()
            if args.alg == 'reverse':
                hg.reverse(params_history[-args.hessian_q-1:], [lambda_x], [inner_opt]*args.hessian_q, loss_outer)
            elif args.alg == 'fixed_point':
                hg.fixed_point(final_params, [lambda_x], args.hessian_q, inner_opt, loss_outer, stochastic=False, tol=tol)
            elif args.alg == 'CG':
                hg.CG(final_params[:len(parameters)], [lambda_x], args.hessian_q, inner_opt_cg, loss_outer, stochastic=False, tol=tol)
            outer_update = lambda_x.grad
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]

        outer_update = torch.squeeze(outer_update)
        with torch.no_grad():
            weight = weight - args.outer_lr*outer_update
        lambda_index_outer = (lambda_index_outer+args.batch_size) % args.training_size

        if args.alg == 'reverse' or args.alg == 'CG' or args.alg == 'fixed_point':
            train_loss_avg = loss_train_avg(train_loader, final_params[0], device, batch_num)
            test_loss_avg = loss_test_avg(test_loader, final_params[0], device)
        else:
            train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
            test_loss_avg = loss_test_avg(test_loader, parameters, device)
        print('Epoch: {:d} Train Loss: {:.4f} Test Loss: {:.4f}'.format(epoch+1, train_loss_avg, test_loss_avg))
        end_time = time.time()

        loss_time_results[epoch+1, 0] = train_loss_avg
        loss_time_results[epoch+1, 1] = test_loss_avg
        loss_time_results[epoch+1, 2] = (end_time-start_time)

        logger.log_value('Train_loss_epoch', train_loss_avg, epoch+1)
        logger.log_value('Test_loss_epoch', test_loss_avg, epoch+1)
        logger.log_value('Train_loss_time', train_loss_avg, int(end_time-start_time))
        logger.log_value('Test_loss_time', test_loss_avg, int(end_time-start_time))
    print(loss_time_results)
    file_name = str(args.seed)+'.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_time_results)

def loss_train_avg(data_loader, parameters, device, batch_num):
    loss_avg, num = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index>= batch_num:
            break
        else:
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            loss = loss_f_funciton(labels, parameters, images)
            loss_avg += loss 
            num += 1
    loss_avg = loss_avg/num
    return loss_avg

def loss_test_avg(data_loader, parameters, device):
    loss_avg, num = 0.0, 0
    for _, (images, labels) in enumerate(data_loader):
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        # images = torch.cat((images, torch.ones(images.size()[0],1)),1)
        labels = labels.to(device)
        loss = loss_f_funciton(labels, parameters, images)
        loss_avg += loss 
        num += 1
    loss_avg = loss_avg/num
    return loss_avg

def gradient_gy(args, labels_cp, parameters, data, weight):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(weight))) + 0.001*torch.pow(torch.norm(parameters),2)
    grad = torch.autograd.grad(loss_regu, parameters, create_graph=True)[0]
    return grad

def gradient_fy(args, labels_cp, parameters, data):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels_cp)
    grad = torch.autograd.grad(loss, parameters)[0]
    return grad

def loss_f_funciton(labels, parameters, data):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels)
    return loss

def nositify(labels, noise_rate, n_class):
    num = noise_rate*(labels.size()[0])
    num = int(num)
    randint = torch.randint(1, 10, (num,))
    index = torch.randperm(labels.size()[0])[:num]
    labels[index] = (labels[index]+randint) % n_class
    return labels

def main():
    args = parse_args()
    print(args)
    train_loader, test_loader = get_data_loaders(args)
    train_model(args, train_loader, test_loader)

if __name__ == '__main__':
    main()
