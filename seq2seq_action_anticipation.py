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
from torch.autograd import Variable
from acticipate import ActicipateDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from  contex_model import *
from movement_model import *
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 



class S2SActionAnticipation(nn.Module):
    
    def __init__(self, data_type, input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5 ):
        super(S2SActionAnticipation, self).__init__()
        self.data_type = data_type
        self.context = ContextModel()
        # self.context.load_state_dict(torch.load("./context_sparcity_denoising.pth"))
        # self.context.eval()

        self.movement = MovementModel()
        # self.movement.load_state_dict(torch.load("./movement_sparcy_denoising.pth"))
        # self.movement.eval()

        self.input_size = input_size #44
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.seq_decode = 5

        self.encode_lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_dim, num_layers = self.n_layers,dropout=rnn_dropout, batch_first=True)

        self.decode = nn.LSTM(self.input_size, self.hidden_dim, self.n_layers,dropout=rnn_dropout, batch_first=True) 
        # dropout layer

        self.dropout = nn.Dropout(fc_dropout)
        self.criterion = nn.NLLLoss()
        

        # linear and sigmoid layers
        self.fc_decode = nn.Linear(self.hidden_dim, 30)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    
    def get_loss(self,output, logits, seq_labels, seq_data, bt):
        seq_data = seq_data.view(-1,seq_data.size(2))
        #seq_data = self.embedding(seq_data)

        mse_ball = ((output[:,:2]-seq_data[:,:2])**2).mean()
        mse_head = ((output[:,2:12]-seq_data[:,2:12])**2).mean()
        mse_mov = ((output[:,12:]-seq_data[:,12:])**2).mean()
        weight_point = 0.5/12
        mse = (10*mse_ball + 2*mse_head)*weight_point +  mse_mov*0.5  
        alpha = 20.0
        
       
        ll = F.log_softmax(logits,2).exp().mean(1).log()
        
        reward = alpha/(bt+1)
        prob,pred = ll.exp().max(1)
        R = ((pred != seq_labels)*reward).unsqueeze(1).float()
        #R = torch.tensor(R)
        J = ((R+1).log()*R).sum(1).mean()
        #mse = ((output-seq_data)**2).mean()
        nll = self.criterion(ll, seq_labels)
        #ent = (ll*ll.exp()).sum(1).mean() #entropy
        return 20*mse + nll + J




    def embedding(self,x):
        xc = x[:,:12]
        xm = x[:,12:]

        if "g" not in self.data_type:
            xc[:,2:] = -1
        if "b" not in self.data_type:
             xc[:,:2] = -1
        if "m" not in self.data_type:
            xm[:,:] = -1       

        xc = self.context(xc)
        xm = self.movement(xm)
        x = torch.cat( (xm,xc),  1) 
        return x

    def forward(self, x, hidden):
        # if hidden is None:
        #     zeros = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device).long()
        #     hidden = (zeros,zeros)

        seq = x.size(1)
        x = x.view(-1,x.size(2))
        emb = self.embedding(x)
        emb = emb.view(-1,seq,emb.size(1))  #B,S,V
        
        encoded, hidden_enc = self.encode_lstm(emb,hidden)
        hidden_dec = hidden_enc
        start = torch.zeros_like(emb)
        outputs = []
        denses = []
        input = start[:,0,...]
        for s in range(self.seq_decode):
            input = input.unsqueeze(1)
            out, hidden_dec = self.decode(input,hidden_dec)
            out = out.view(-1, out.size(2))
            denses.append(out.unsqueeze(1))
            out = self.fc_decode(out)
            outputs.append(out.unsqueeze(1))
            input = self.embedding(out)
        
        out = torch.cat(outputs, 1)
        out = out.contiguous().view(-1, out.size(2))
        denses = torch.cat(denses, 1)
        denses = denses.contiguous().view(-1, denses.size(2))
        denses = self.dropout(denses)
        logits = self.fc(denses)
        #out = out.view(x.size(0), -1)[:, -1]  # get last batch of labels
        
        return out, logits, hidden_enc






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
        input_size = 64
        output_size = 12
        hidden_dim = np.random.choice(a= [parse.hidden_dim],size=1)[0]
        n_layers =  np.random.choice(a= [parse.n_layers],size=1)[0]
        rnn_dropout = np.random.choice(a= [0.5],size=1)[0]
        fc_dropout = np.random.choice(a= [.7],size=1)[0]
    
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
        test_results = []
        for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=folds)): 
            model = S2SActionAnticipation(parse.data_type, input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.8, fc_dropout = 0.5)
            model = model.to(device)
            model = self.train(args,model, train_data, max_clip_sequence, device, logging)  
            acc = self.predict(model, test_data,test_results, logging)
            accuracy[k] = acc
        

        pickle.dump(test_results, open( "prediction_s2s_{}.pkl".format("_".join(parse.data_type)), "wb" ), protocol=2)
            
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
            input_size = 64
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
            model = S2SActionAnticipation(input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5)
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
        padded_data = np.ones((len(videos),max,44))*-1
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

    def to_batch(self,videos, batch_size = 32, seq = 1, max = 100, decode_seq = 5):
            indexes = list(range(len(videos)))
            random.shuffle(indexes)
            videos = [videos[i] for i in indexes]
            for b in range(len(videos)//batch_size):
                video_batch = [videos[i] for i in range(b*batch_size,(b+1)*batch_size)]
                padded_data, padded_labels = self._padding(video_batch,max= max)
                size = (padded_data.shape[1]-decode_seq) // seq
                for s in range(size):
                    label = padded_labels[:,(s+1)*seq:(s+1)*seq+decode_seq]
                    data = padded_data[:,s*seq:(s+1)*seq+decode_seq]
                    data = data[:,:,[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
                    yield data , label , True if s == (size-1) else False

    def predict(self, model, videos, results, log):
        num_class = 12
        log.info("Predicting...")
        model.cpu()
        model.eval()
        criterion = nn.NLLLoss()
        running_loss = 0
        running_corrects = 0
        total = 1e-18 
        for (video, interval) in videos:
            hidden = None
            probs = np.zeros((len(video),num_class))
            for i, data in enumerate(video):
                label = torch.tensor(data[0]-1).unsqueeze(0).long()
                data= data[1:]
                data = data[[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
                data = torch.from_numpy(data).float()
                data = data.unsqueeze(0).unsqueeze(0)
                _, out, hidden = model(data,hidden)
                hidden = ([h.data for h in hidden])
                prob = F.log_softmax(out,1).exp().mean(0).detach().numpy()
                probs[i] = prob
                loss = criterion(F.log_softmax(out,1).exp().mean(0).log().unsqueeze(0), label)
                running_loss += loss.item()
            
            pred = np.argmax(prob)
            label = label.data.numpy()[0]
            total += 1

            if pred != label:
                print(pred,label)
            results.append({"pred":pred, "label":label,  "probs":probs, "interval":interval})
            running_corrects += 1*(pred == label)

        log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/total, running_corrects/float(total)*100))
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
        decode_seq = 5
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        indexes = list(range(len(train_data)))
        steps = len(train_data)//batch
        accuracies = []
        for epoch in range(epochs):
            scheduler.step()
            random.shuffle(indexes)
            train_data = [ train_data[ i] for i in indexes]
            running_loss = 0
            running_corrects = 0 
            total = 1e-18
            hidden = None
            probs = []
            bt = 0.
            for data, label,end_batch in self.to_batch(train_data,seq=sequence, batch_size = batch, max =max_clip_sequence):
                bt += 1.
                #log.info("batch {}/{}".format(bt,steps))
                data, seq_data = data[:,:-decode_seq], data[:,-decode_seq:]
                data,label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).to(device)
                seq_data = torch.from_numpy(seq_data).float().to(device)

                emb, out,  hidden = model(data,hidden)
                #out = out.view(-1,out.size(2))
                label = label.contiguous().view(-1)
                hidden = ([h.data for h in hidden])
                out = out.view(-1, decode_seq,out.size(1))
                label = label.view(-1, decode_seq)[:,-1]
                loss = model.get_loss(emb, out, label, seq_data,(bt*sequence)) #(bt*sequence/max_clip_sequence)
                
                #loss = criterion(F.log_softmax(out,1), label) 

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if end_batch:
                    out = out[:,-1,:]
                    hidden = None
                    bt = 0.
                    #loss = criterion(F.log_softmax(out,1), label)
                    prob, preds = torch.max(out, 1)
                    probs.append(prob.cpu().detach().numpy())
                    total += out.size(0)
                    running_loss += loss.item() * out.size(0)
                    running_corrects += torch.sum(preds == label.data).double()
            probs = np.array(probs)
            time = np.arange(len(probs))
            
            acc = running_corrects/float(total)
            accuracies.append(acc)
            
            if len(accuracies)  > 10: 
                del accuracies[0]
                mean = sum(accuracies)/float(len(accuracies))
                if  mean > 0.99: break
            #print(len(probs))
            # plt.plot(probs)
            # plt.show()
            log.info("---> Epoch{}  loss = {}    accuracy = {:.2f}%".format(epoch + 1, running_loss/total, acc*100) )
            
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
            print("Sending kill to threads...")
            for t in threads:
                t.kill = True
            sys.exit(1) 
    

            
if __name__ == "__main__":
     optmize()

#100 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --grad_clip 4.7
#99.58 python action_anticipation.py --seq 120 --batch_size 200 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .6
#98.75 python action_anticipation.py --seq 32 --batch_size 32 --hidden_dim  256 --lr 1e-3 --epoch 200 --n_layers 4 --trunc_seq 128 --grad_clip 5
#ball 95.42 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .5
# remain python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip 5
# 6 class python action_anticipation.py --seq 120 --batch_size 108 --hidden_dim  32 --lr 2e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --data_type g --grad_clip 0.15