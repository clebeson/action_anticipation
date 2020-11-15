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



class ActAnticipationModel(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5 ):
        super(ActAnticipationModel, self).__init__()
        self.input_size = input_size #44
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers,dropout=rnn_dropout, batch_first=True) 
        # dropout layer
        self.dropout = nn.Dropout(fc_dropout)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(self.hidden_dim, self.output_size)


    def forward(self, x, hidden):

        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        #out = out.view(x.size(0), -1)[:, -1]  # get last batch of labels
        
        return out, hidden






class optmize:     
    def __init__(self,id=0,file_id=0, args = None):
        self.run_experiment()


       
    def run_experiment(self):
        class param:pass
        #param_msg = "input_size = {},output_size = {},hidden_dim = {},n_layers = {},rnn_dropout = {},fc_dropout = {},epoch= {},lr = {},batch = {},sequence = {},clip = {}, max={}"
        args = param()
        parse = utils.parse_args()
        #logger = logging.getLogger('Manual_exp')
        logging.basicConfig(level=logging.INFO)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset = ActicipateDataset()
        input_size = 44
        output_size = 12
        hidden_dim = np.random.choice(a= [parse.hidden_dim],size=1)[0]
        n_layers =  np.random.choice(a= [parse.n_layers],size=1)[0]
        rnn_dropout = np.random.choice(a= [0.3],size=1)[0]
        fc_dropout = np.random.choice(a= [.5],size=1)[0]
    
        #train params
        lr = np.random.choice(a= [parse.lr],size=1)[0]
        batch = np.random.choice(a= [parse.batch_size],size=1)[0]
        sequence = np.random.choice(a= [parse.seq],size=1)[0]
        clip = np.random.choice(a= [parse.grad_clip],size=1)[0]
        max_clip_sequence = np.random.choice(a= [parse.trunc_seq],size=1)[0]
        
        epoch = parse.epoch
        args.hidden_dim = hidden_dim
        args.n_layers = n_layers
        args.rnn_dropout = rnn_dropout
        args.fc_dropout = fc_dropout
        args.epoch = epoch
        args.lr = lr
        args.batch_size = batch
        args.seq = sequence
        args.clip = clip

        #logging.info(param_msg.format(input_size,output_size,hidden_dim,n_layers,rnn_dropout,fc_dropout,epoch,lr,batch,sequence,clip,max_clip_sequence))
        folds = 10
        accuracy = np.zeros((folds))
        for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=folds)): 
            model = ActAnticipationModel(input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5)
            model = model.to(device)
            model = self.train(args,model, train_data, max_clip_sequence, device, logging)  
            acc = self.predict(model, test_data, logging)
            accuracy[k] = acc
            
        logging.info("Cross val: {:.2f}%".format(accuracy.mean()*100))

       


    def find_parameters(self):
        class param:pass
        args = param()


        logger = logging.getLogger('simple_example{}'.format(file_id))
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('log_test{}.log'.format(file_id))
        fh.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s :: %(message)s')
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)

        param_msg = "input_size = {},output_size = {},hidden_dim = {},n_layers = {},rnn_dropout = {},fc_dropout = {},epoch= {},lr = {},batch = {},sequence = {},clip = {}, max={}"
        
        #logger.basicConfig(filename="log_test_{}.txt".format(file_id), filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s')
        device = torch.device('cuda:{}'.format(id) if torch.cuda.is_available() else 'cpu')
        logger.info("\n\n\nRunning in {}".format(device))

        dataset = ActicipateDataset()
        best_args = None
        best_acc = 0
        for i in range(20):
        #model params
            input_size = 44
            output_size = 12
            hidden_dim = np.random.choice(a= [128,256,512],size=1)[0]
            n_layers =  np.random.choice(a= [2,4],size=1)[0]
            rnn_dropout = np.random.choice(a= [.5],size=1)[0]
            fc_dropout = np.random.choice(a= [.5],size=1)[0]
            
            #train params
            epoch = 20
            lr = np.random.choice(a= [1e-3,5e-3,5e-4],size=1)[0]
            batch = np.random.choice(a= [16,32,64],size=1)[0]
            sequence = np.random.choice(a= [10,16,32],size=1)[0]
            clip = np.random.choice(a= [3,5,10],size=1)[0]
            max_clip_sequence = np.random.choice(a= [80,100],size=1)[0]
            
            args.hidden_dim = hidden_dim
            args.n_layers = n_layers
            args.rnn_dropout = rnn_dropout
            args.fc_dropout = fc_dropout
            args.epoch = epoch
            args.lr = lr
            args.batch_size = batch
            args.seq = sequence
            args.clip = clip

            logger.info(param_msg.format(input_size,output_size,hidden_dim,n_layers,rnn_dropout,fc_dropout,epoch,lr,batch,sequence,clip,max_clip_sequence))
            model = ActAnticipationModel(input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5)
            accuracy = np.zeros((5))
            for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=5)): 
                model = model.to(device)
                model = self.train(args,model, train_data, max_clip_sequence, device, logger)  
                acc = self.predict(model, test_data, logger)
                accuracy[k] = acc
                if acc < 0.9:break
            mean_acc = accuracy.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_args = args

        logger.info("Best Acc: {}".format(best_acc))    
        logger.info(param_msg.format(input_size,output_size,best_args.hidden_dim,\
                    best_args.n_layers,best_args.rnn_dropout,best_args.fc_dropout,best_args.epoch,\
                    best_args.lr,best_args.batch,best_args.sequence,best_args.clip,best_args.max_clip_sequence))




    def _padding(self,videos,max):
        sizes = []
        [sizes.append(len(v)) for v in videos]
        sizes = np.array(sizes)
        padded_data = np.zeros((len(videos),max,44))
        padded_labels = np.ones((len(videos),max))*-1
        
        for i,video in enumerate(videos):
            padded_labels[i] = video[0,0]-1
            if len(video) > max:
                video = video[:max]
            padded_data[i,:len(video)] = video[:,1:]
            #padded_data[i,-len(video):] = video[-1,1:]
           # print(padded_data[i].sum()- video[:,1:].sum())

        padded_data = padded_data.astype(float)
        padded_labels = padded_labels.astype(int)
        return padded_data, padded_labels

    def to_batch(self,videos, batch_size = 32, seq = 1, max = 100):
            indexes = list(range(len(videos)))
            random.shuffle(indexes)
            videos = [videos[i] for i in indexes]
            for b in range(len(videos)//batch_size):
                video_batch = [videos[i] for i in range(b*batch_size,(b+1)*batch_size)]
                padded_data, padded_labels = self._padding(video_batch,max= max)
                size = padded_data.shape[1] // seq
                for s in range(size):
                    label = padded_labels[:,s*seq:(s+1)*seq]
                    data = padded_data[:,s*seq:(s+1)*seq]
                    yield data , label , True if s == (size-1) else False

    def predict(self, model, videos, log):
        log.info("Predicting...")
        model.cpu()
        model.eval()

        criterion = nn.NLLLoss()
        running_loss = 0
        running_corrects = 0
        total = 1e-18 
        for video in videos:
            hidden = None
            for data in video:
                label = torch.tensor(data[0]-1).long()
                data = torch.from_numpy(data[1:]).float()
                data = data.unsqueeze(0).unsqueeze(0)
                out, hidden = model(data,hidden)
                hidden = ([h.data for h in hidden])
            label = label.unsqueeze(0)
            loss = criterion(F.log_softmax(out,1), label)
            label = label.contiguous().view(-1)
            prob, preds = torch.max(out, 1)
            total += out.size(0)
            running_loss += loss.item() * out.size(0)
            running_corrects += torch.sum(preds == label.data).double()

        log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/total, running_corrects/float(total)*100) )
        return running_corrects/float(total)
    

            


    def train(self, args, model, train_data,max_clip_sequence,device, log):
        model.train()
        log.info("Trainning...")
        #Hyperparameters
        epochs = args.epoch #30
        batch = args.batch_size#32
        sequence = args.seq #10
        clip = args.clip #5
        lr=args.lr#1e-3
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        indexes = list(range(len(train_data)))
        steps = len(train_data)//batch
        for epoch in range(epochs):
            scheduler.step()
            random.shuffle(indexes)
            train_data = [ train_data[ i] for i in indexes]
            running_loss = 0
            running_corrects = 0 
            total = 0
            hidden = None
            probs = []
            bt = 0
            for data, label,end_batch in self.to_batch(train_data,seq=sequence, batch_size = batch, max =max_clip_sequence):
                bt += 1
                #log.info("batch {}/{}".format(bt,steps))
                data,label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).to(device)
                #print(data[0].sum())
                out, hidden = model(data,hidden)
                #out = out.view(-1,out.size(2))
                label = label.contiguous().view(-1)
                hidden = ([h.data for h in hidden])
                loss = criterion(F.log_softmax(out,1), label)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if end_batch:
                    out = out.view(-1, sequence,out.size(1))[:,-1,:]
                    label = label.view(-1, sequence)[:,-1]
                    loss = criterion(F.log_softmax(out,1), label)

                    prob, preds = torch.max(out, 1)
                    probs.append(prob.cpu().detach().numpy())
                    total += out.size(0)
                    running_loss += loss.item() * out.size(0)
                    running_corrects += torch.sum(preds == label.data).double()
            probs = np.array(probs)
            time = np.arange(len(probs))
            #print(len(probs))
            # plt.plot(probs)
            # plt.show()
            if running_corrects/float(total) > 0.99: break
            #log.info("---> Epoch{}  loss = {}    accuracy = {:.2f}%".format(epoch + 1, running_loss/total, running_corrects/float(total)*100) )
        return model
        
def opt(id=0,file_id=0):
    optmize(id,file_id)



def has_live_threads(threads):
    return True in [t.isAlive() for t in threads]

def main():
    threads = []
    for index in [[0,0],[1,1],[2,2],[0,3],[1,4],[2,5]]:
        x = threading.Thread(target=opt, args=tuple(index))
        threads.append(x)
        x.start()

    while has_live_threads(threads):
        try:
            # synchronization timeout of threads kill
            [t.join(1) for t in threads
             if t is not None and t.isAlive()]
        except KeyboardInterrupt:
            # Ctrl-C handling and send kill to threads
            print "Sending kill to threads..."
            for t in threads:
                t.kill = True
            sys.exit(1) 
    

            
if __name__ == "__main__":
     optmize()
