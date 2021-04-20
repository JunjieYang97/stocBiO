#!/usr/bin/env python3

"""
"""

import os
import random

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision as tv
from torch.autograd import grad

import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

from statistics import mean
from copy import deepcopy

import pickle

class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def task_adapt(loss, model, lr):
    try:
        gradients = grad(loss, model.parameters())
    except RuntimeError:
        traceback.print_exc()
    
    if gradients is not None:
        params = list(model.parameters())
        if not len(gradients) == len(list(params)):
            msg = 'WARNING:Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(gradients)) + ')'
            print(msg)
        for p, g in zip(params, gradients):
            p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    model._apply(lambda x: x)
    
def fast_adapt(batch,
               head_dim,
               features,
               loss,
               fast_lr,
               reg_lambda,
               adaptation_steps,
               shots,
               ways,
               device=None):
                   
    head = torch.nn.Linear(head_dim, ways)
    head.to(device)

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # print('before_data_size=' + str(data.size(0)))
    
    data = features(data)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # print('data_size=' + str(data.size(0)))
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        train_error = loss(head(adaptation_data), adaptation_labels) 
        task_adapt(train_error, head, fast_lr)

    predictions = head(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    del head
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.001, 
        fast_lr=0.1,   # original 0.1
        reg_lambda=0,
        adapt_steps=5, # original: 5
        meta_bsz=32,
        iters=1000, # orginal: 1000
        cuda=1,
        seed=42,
):
    
    print('hlr='+str(meta_lr)+' flr='+str(fast_lr)+' reg='+str(reg_lambda))
    
    cuda = bool(cuda)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Datasets
    train_dataset = l2l.vision.datasets.FC100(root='~/data',
                                              transform=tv.transforms.ToTensor(),
                                              mode='train')
    valid_dataset = l2l.vision.datasets.FC100(root='~/data',
                                              transform=tv.transforms.ToTensor(),
                                              mode='validation')
    test_dataset = l2l.vision.datasets.FC100(root='~/data',
                                              transform=tv.transforms.ToTensor(),
                                             mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        FusedNWaysKShots(train_dataset, n=ways, k=2*shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=ways, k=2*shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        FusedNWaysKShots(test_dataset, n=ways, k=2*shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)


    # Create model
    features = l2l.vision.models.ConvBase(output_size=64, channels=3, max_pool=True)
    features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, 256)))
    features.to(device)
    head_dim = 256
    
    # Setup optimization
    all_parameters = list(features.parameters())
    
    optimizer = torch.optim.Adam(all_parameters, lr=meta_lr)
    # optimizer = torch.optim.SGD(all_parameters, lr=meta_lr)
    
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    training_accuracy =  torch.ones(iters)
    test_accuracy =  torch.ones(iters)
    running_time = np.ones(iters)
    import time
    start_time = time.time()

    for iteration in range(iters):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        
        for task in range(meta_bsz):
            # Compute meta-training loss
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               head_dim,
                                                               features,
                                                               loss,
                                                               fast_lr,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               head_dim,
                                                               features,
                                                               loss,
                                                               fast_lr,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               head_dim,
                                                               features,
                                                               loss,
                                                               fast_lr,
                                                               reg_lambda,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
        
        training_accuracy[iteration] = meta_train_accuracy / meta_bsz
        test_accuracy[iteration] = meta_test_accuracy / meta_bsz
        

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        print('Meta Valid Error', meta_valid_error / meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
        print('Meta Test Error', meta_test_error / meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_bsz)
        
        optimizer.step()
        end_time = time.time()
        running_time[iteration] = end_time - start_time
        print('time per iteration', end_time - start_time)
        
    
    return training_accuracy.numpy(),test_accuracy.numpy(), running_time


if __name__ == '__main__':
    train_accuracy = []
    test_accuracy = []
    run_time = []
    
    seeds = [42,52,62,72,82]
    stp = 20
    lr = 0.001
    fastlr=0.1
    reg=0
    
    for seed in seeds: 
        training_accuracy,testing_accuracy, running_time = main(meta_lr=lr, 
                                                                adapt_steps=stp, 
                                                                fast_lr=fastlr, 
                                                                reg_lambda=reg, 
                                                                iters=1500, 
                                                                seed=seed)
        train_accuracy.append(training_accuracy)
        test_accuracy.append(testing_accuracy)
        run_time.append(running_time)
    
    # save 
    # from datetime import datetime
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    
    pstr = '_lr_' + str(lr) + '_fastlr_' + str(fastlr) + '_steps_' + str(stp)

    with open('exp_data/train_accuracy' + pstr, 'wb') as f:
        pickle.dump(train_accuracy, f)
        
    with open('exp_data/test_accuracy' + pstr, 'wb') as f:
        pickle.dump(test_accuracy, f)
        
    with open('exp_data/run_time' + pstr, 'wb') as f:
        pickle.dump(run_time, f)
        
    