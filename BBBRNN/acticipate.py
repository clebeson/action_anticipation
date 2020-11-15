from collections import deque
import numpy as np
import cv2
import csv
import os
import argparse
import math
import pandas as pd
import glob
import pickle
import random


class ActicipateDataset:

    def __init__(self, classes = None ):
        self.data = np.load("labels_normalized.npy").astype(float)

        self.classes = list(range(12)) if classes is None else classes
        self.features = list(range(len(self.data[0]-2)))
        values = self.data[:,2:]
        values[values<0] = 0
        mean = values.mean(1, keepdims = True)
        std = values.std(1,  keepdims = True)
        self.data[:,2:] = (values - mean) / (std+1e-18)

    def decode(self, classes, features):
        self.classes = classes
        self.features = features
        self._decode_data()


    def get_test(self):
        return [(self.data[b:e+1,self.features], int(self.data[b,1]-1), (b,e)) for b,e in self.test]
    
    def get_train(self):
        return [(self.data[b:e+1,self.features], int(self.data[b,1]-1)) for b,e in self.train]
    
    def _decode_data(self):
        name = "holdout{}.pkl".format("" if len(self.classes) == 12 else 6)
        if os.path.isfile(name):
            with open(name,"rb") as file:
                f = pickle.load(file)
                self.train, self.test = f["train"], f["test"]
            return None

   
        b,e = 0,0
        prev = int(self.data[0,1])
        videos = []
        count = 0
        
        for d in self.data:
            if d[1] != prev:
                e = int(d[0]-1)
                #print("{}-{}-{} - {}".format(prev,b,e, e-b))
                #if prev not in [7,8,9,10,11,12]:
                videos.append([b,e])
                prev = int(d[1])
                b = int(d[0])
                count += 1
        e = int(d[0])
        #if prev not in [7,8,9,10,11,12]:
        videos.append([b,e])
        self.data = self.data.astype(float)
        videos = np.array(videos).astype(int)
        labels = np.array([int(self.data[b,1]-1) for b,_ in videos])
        self.train, self.test = self._stractfied_data(labels)
        self.train, self.test = videos[self.train], videos[self.test]
        print(len(videos))


    def _stractfied_data(self,labels):
        indices = np.array(list(range(len(labels))))
        test = []
        train = []
        for l in self.classes :#np.unique(labels):
            values = indices[labels == l]
            np.random.shuffle(values)
            cut = int(0.2*len(values))
            test += values[:cut].tolist()
            train += values[cut:].tolist()
        return train, test



    # def holdout(self, rounds=1, train_percents = 0.8):
    #     rounds=1 if rounds < 1 else rounds
    #     train_percents = 0.8 if train_percents>1 or train_percents <0 else train_percents
    #     indexes = list(range(len(self.videos)))
    #     for round in range(rounds):
    #         random.shuffle(indexes)
    #         cut = int(train_percents*240)
    #         train_indexes = indexes[:cut]
    #         test_indexes = indexes[cut:]
            
    #         yield [self.data[b:e+1,1:] for b,e in self.videos[train_indexes]], [self.data[b:e+1,1:] for b,e in self.videos[test_indexes]]
        

    def cross_validation(self, k=5):
        if not hasattr(self, 'train'):
            self._decode_data()
        #k = 5 if k > 10 or k<1 else k
        fold_size = len(self.train)//k
        indexes = list(range(len(self.train)))
        random.shuffle(indexes)
        for fold in range(k):
            begin = fold*fold_size
            end = begin+fold_size
            if fold == k-1: end = len(self.train)
            val_indexes = indexes[begin:end]
            train_indexes = [index for index in indexes if index not in val_indexes]
            train = [(self.data[b:e+1,self.features], int(self.data[b,1]-1)) for b,e in self.train[train_indexes]]
            val = [(self.data[b:e+1,self.features], int(self.data[b,1]-1),(b,e)) for b,e in self.train[val_indexes]]
            yield  train, val





if __name__ == "__main__":
    act = ActicipateDataset()
    act.classes = list(range(6))
    act._decode_data()
    pickle.dump({"train":act.train,"test":act.test}, open("holdout6.pkl","wb"))
    



