from collections import deque
import numpy as np
import cv2
import csv
import sys
import os
import argparse
import imutils
import math
import pims
import pandas as pd
from moviepy.editor import *
import glob
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from acticipate import ActicipateDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 




class ContextModel(nn.Module):
    def __init__(self ):
        super(ContextModel, self).__init__()
        h, b= 10, 2
        self.dropout = 0.2
        self.dp_training = True
        self.encoder_ball_position = nn.Sequential(nn.Linear(b, 16), nn.ReLU())
        self.encoder_head = nn.Sequential(nn.Linear(h, 16), nn.ReLU())
        self.combine = nn.Sequential(nn.Linear(32, 16),  nn.ReLU())

    def forward(self, x):
        ball = x[...,:2]
        head = x[...,2:]

        ball = self.encoder_ball_position(ball)
        ball = F.dropout(ball,self.dropout, training=self.dp_training)

        head = self.encoder_head(head)
        head = F.dropout(head,self.dropout, training=self.dp_training)

        x = torch.cat((ball, head), -1)
        encode = self.combine(x)

        return encode


    def get_kl(self):
        rho = torch.FloatTensor([0.01]*32).unsqueeze(0).cuda()
        rho_hat = torch.sum(self.encoded, dim=0, keepdim=True)
        p = F.softmax(rho, dim=1)
        q = F.softmax(rho_hat, dim =1)
        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        
        return s1 + s2
    
    def add_noise(self,data):
            noise = data.data.new(data.size()).normal_(0, 0.01)
            return data + noise


def augment(batch,thres = 0.4):
    values = [[0, 1], [2, 3,4,5,6,7,8,9,10,11,12,13] ]
    for i in range(len(batch)):
        if np.random.randn() < thres:
            size = np.random.randint(1,2,1)
            change = np.random.choice(a = [0,1],size = size)
            for c in change:                    
                batch[i,values[c] ] = -1
    return batch

def next_batch(data, batch = 32, aug = False):
    indexes = list(range(len(data)))
    random.shuffle(indexes)
    data = data[indexes]

    steps = len(indexes)//batch
    for i in range(steps):
        b = i*batch
        e = b+batch    #ball     #head  and head
        b_data = data[b:e,[0, 1,  2,3,30,31,32,33,34,35,36,37, 42, 43]]
        if aug: b_data = augment(b_data)
        yield b_data
    

def predict(model,test):
    model.eval()
    epoch_loss = 0
    total = 0
    for batch_data in next_batch(test,batch = 30, aug=False):
            input = torch.from_numpy(batch_data).float().cuda()
            # ===================forward=====================
            output = model(input)
            loss = criterion(output, input)
            # ===================backward====================
            epoch_loss += loss.item()*output.size(0)
            total += output.size(0)
        # ===================log========================
    out = output.cpu().detach().numpy()
    input = input.cpu().detach().numpy()
    for o,i in zip(out,input):
       print([(j,k,j-k) for j,k in zip(i,o)])
    print('Predict loss:{:.6f}'.format( epoch_loss))
            

if __name__ == "__main__":

    num_epochs = 200
    batch_size = 32
    learning_rate = 5e-3
    model = ContextModel().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    dataset = ActicipateDataset()
    data = dataset.data
    data = np.concatenate((np.zeros((2000,44)),data[:,2:]), 0)
    size = len(data)
    indexes = list(range(size))
    random.shuffle(indexes)
    data = data[indexes]
    test = data[:int(size*0.4)]
    train = data[int(size*0.4):]
    # model.load_state_dict(torch.load("./context_autoencoder.pth"))
    # model.cuda()
    # predict(model,test)
    # return
    for epoch in range(num_epochs):
        scheduler.step()
        epoch_loss = 0
        total = 0
        for batch_data in next_batch(train, batch_size, aug= True):
            input = torch.from_numpy(batch_data).float().cuda()
            # ===================forward=====================
            output = model(input)
            loss = criterion(output, input) +  0.01*model.get_kl()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*output.size(0)

        # ===================log========================
        print('epoch [{}/{}], loss:{:.6f}'
            .format(epoch + 1, num_epochs, epoch_loss))
        
    predict(model,test)
    torch.save(model.state_dict(), './context_sparcity_denoising.pth')