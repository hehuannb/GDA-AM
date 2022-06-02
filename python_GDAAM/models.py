#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:28:38 2021

@author: huan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import datasets

class Generator(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=2):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, int(hidden_size))
        self.map3 = nn.Linear(hidden_size, int(hidden_size))
        self.map4 = nn.Linear(int(hidden_size), output_size)
        self.f = torch.relu
    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        x = self.f(x)
        return self.map4(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = torch.sigmoid
        self.f2 = torch.tanh
        
    def forward(self, x):
        x = self.f2(self.map1(x))
        x = self.f2(self.map2(x))
        return self.f(self.map3(x))
    
        
def build_dataloader(batch_size, DATASET ='8gaussians'):  # mix of 8 Gaussians (https://github.com/igul222/improved_wgan_training)

        
        
    if DATASET == '25gaussians':
        dataset = []
        for i in range(int(100000 / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(int(len(dataset) / batch_size)):
                yield dataset[i * batch_size:(i + 1) * batch_size]

    elif DATASET == 'swissroll':

        while True:
            data = datasets.make_swiss_roll(
                n_samples=batch_size,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  
            yield data

    elif DATASET == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        sigs = [np.eye(2) * 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,
                np.eye(2)* 1e-2,]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for _ in range(int(100000/8)):
            for center in range(8):
                mu_, sig_ = centers[center], sigs[center]
                data = np.random.multivariate_normal(mu_, sig_)
                dataset.append(data)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        while True:
            for i in range(int(len(dataset) / batch_size)):
                yield dataset[i * batch_size:(i + 1) * batch_size]
            
        