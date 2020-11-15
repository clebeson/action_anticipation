from hmmlearn import hmm

import numpy as np 
from acticipate import *
import pickle
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import hyperopt
import torch
import torch.nn as nn
from torch.optim import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import math
np.random.seed(42)


class ConvModel(nn.Module):
    def __init__(self,params):
        super(ConvModel, self).__init__()
        max = params["max_seq"]*4
        self.conv = nn.Sequential(
            nn.Conv1d(1, params["out_ch1"], params["kernel1"], stride=1, bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2),
            nn.Conv1d(params["out_ch1"], params["out_ch2"], params["kernel2"], stride=1,bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2),
            nn.Conv1d(params["out_ch2"], params["out_ch3"], params["kernel3"], stride=1, bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2)
        )

        for k in [params["kernel1"],params["kernel2"],params["kernel3"]]:
            n = (max - k)/2.
            max = n if int(n) == n else int(n)+1

        self.clf = nn.Sequential(nn.Linear(int(params["out_ch3"]*max), params["hidden_layer"], bias = True),
                 nn.ReLU(), nn.Linear( params["hidden_layer"], 12), nn.LogSoftmax(1))
        
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.clf(x)


def get_model(name):
    train_clf = NB_train if model == "NB" \
                else hmm_train \
                    if model =="HMM" \
                        else MLP_train\
                            if model =="MLP" \
                                else CONV_train
    test_clf = NB_test if model == "NB" \
        else hmm_test \
            if model =="HMM" \
                else MLP_test\
                    if model =="MLP" \
                        else CONV_test
    return train_clf, test_clf

def get_hyperparameters(model):
    if model == "NB":
        return {
            "max_seq": hp.choice('max_seq',[32,36,40,48,60,64,72]),
            
        }
    elif model =="HMM":
        return {
            "max_seq": hp.choice('max_seq',[30]), 
            "states":hp.choice('states', [2,3,4,5,6]),
        }
    elif model =="MLP":
        return  {
            "max_seq": hp.choice('max_seq',[32,36,40,48,60,64,72]), 
            "hidden_dim":hp.choice('hidden_dim',[(16,), (32,), (48,), (64,), (96,), (128,), (256,), (512,)]), 
            "lr":hp.choice('lr', [1e-1,1e-2,1e-3])
        }
     
    return  {
        "max_seq": hp.choice('max_seq',[32,36,40,48,60,64,72]), 
        "batch": hp.choice('batch',[4,8,16,32,64]), 
        "out_ch1":hp.choice('out_ch1',[8,16,32,64]), 
        "out_ch2":hp.choice('out_ch2',[8,16,32,64]), 
        "out_ch3":hp.choice('out_ch3',[8,16,32,64]), 
        "kernel1":hp.choice('kernel1',[3,5,7,9]), 
        "kernel2":hp.choice('kernel2',[3,5,7,9]), 
        "kernel3":hp.choice('kernel3',[3,5,7,9]), 
        "hidden_layer":hp.choice('hidden_layer',[16,32,64,128,256]), 
        "lr":hp.choice('lr', [1e-1,1e-2,1e-3,1e-4]),
    }


def stractfied_data(labels):
    indices = np.array(list(range(len(labels))))
    test = []
    train = []
    for l in np.unique(labels):
        values = indices[labels == l]
        np.random.shuffle(values)
        cut = int(0.2*len(values))
        test += values[:cut].tolist()
        train += values[cut:].tolist()
    return train, test


def fixed_sequece_size(seq, max):
    sequences = np.zeros((max,len(seq[0])))-1
    values  = seq[:max] if len(seq) >= max else seq
    sequences[:len(values)] = values
    return sequences

def bayesian_optimization(params):
        train = params["data"]
        model = params["model"]
        train_indices = list(range(len(train)))
        np.random.shuffle(train_indices)
        cross_acc = 0
        k = 10
        fold_size = len(train_indices)//k

        results = []
        train_clf, test_clf = get_model(model)
        #cross validaation
        for fold in range(k):
           
            begin = fold*fold_size
            end = begin+fold_size
            if fold == k-1: end = len(train_indices)
            val_fold_idx = train_indices[begin:end]
            train_folds_idx = [idx for idx in train_indices if idx not in val_fold_idx]
            train_folds = [train[idx] for idx in train_folds_idx]
            val_fold    = [train[idx] for idx in val_fold_idx]
            model = train_clf(train_folds, params)
            acc = test_clf(model,val_fold,params["max_seq"],results)

            # print ("Acc fold {} = {:.2f} ".format(fold,100*acc))
            cross_acc += acc
        cross_acc/=k
        p = {}
        for key in params.keys():
            if key =="data" or key == "model":continue
            p[key] = params[key]

        # print("cross acc {:.2f}".format(cross_acc*100))
        return {'loss': 1-cross_acc, 'status': STATUS_OK, 'params': p}



def get_observation(seq, t, max):
    sequences = np.zeros((max,len(seq[0])))-1
    if t >=max: t=max
    values  = seq[:t] if len(seq) >= t else seq
    sequences[:len(values)] = values
    return sequences


def viterbi_approximation(model, observation):
        A = model.transmat_
        m = model.means_
        cv = model.covars_
       
        states = model.predict(observation)
        # print(S.shape)
        B = model.predict_proba(observation)
        v = model.startprob_.tolist()[states[0]]
        v *= multivariate_normal(m[states[0]],cv[states[0]]).pdf(observation[0])
        for i, (o, s_i,s_j) in enumerate(zip(observation[1:], states[:-1],states[1:])):
            b = multivariate_normal.pdf(o,m[s_j],cv[s_j])
            v *= A[s_i,s_j]*b
        return v

def softmax(x, axis= 1):
    """Compute softmax values for each sets of scores in x."""
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def hmm_train(train, params):
    ste, seq = params["states"], params["max_seq"]
    print(f"{ste}  -  {seq}")
    models = [hmm.GaussianHMM(n_components=params["states"],  verbose=False, tol = 0.01, n_iter = 30) for _ in range(12)]
    # for model in models:
    #     model.transmat_ = np.ones((params["states"],params["states"]))/params["states"]

    sequencies = [[] for _ in range(12)]
    lengths = [[] for _ in range(12)]

    for d,l in train:
        sequencies[l] += fixed_sequece_size(d,params["max_seq"]).tolist()
        lengths[l] += [params["max_seq"]]#[e-b]

    sequencies = np.array(sequencies)
    lengths = np.array(lengths)
    for i, (s, l) in enumerate(zip(sequencies,lengths)):
        models[i] = models[i].fit(s,l)
    return models

def hmm_test(models, test,max_seq, results):
    predictions = np.zeros((len(test),12))
    
    labels = []
    for n, (d,l) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            obs_pred = np.zeros(12)
            observation = get_observation(d,t,max_seq) #d[:t]
            for i,model in enumerate(models):
                score = model.score(observation)
                obs_pred[i] = score #viterbi_approximation(model, observation)
            obs_pred = obs_pred - obs_pred.min() #(obs_pred - obs_pred.min())/(obs_pred.max() - obs_pred.min()+1e-18)
            obs_pred[obs_pred==0] = 1e-6
            obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)
            #     video_preds.append(obs_pred)
            video_preds.append(obs_pred)

        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
   
    return  acc


def NB_train(train, params):
    model = GaussianNB()
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model







def NB_test(model, test, max_seq, results):
    predictions = np.zeros((len(test),12))
    labels = []
    for n, (d,l) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            observation = get_observation(d,t,max_seq).reshape(1,-1) #d[:t]
            obs_pred = model.predict_proba(observation)[0]
            # obs_pred[obs_pred==0] = 1e-6
            # obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)

            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc

def CONV_train(train, params):
    model = ConvModel(params)
    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=params["lr"])
    batch_size = params["batch"]
    indices = list(range(len(train)))
    steps = len(indices)//batch_size
    sequencies = []
    labels = []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    for epoch in range(50):
        np.random.shuffle(indices)
        corrects = 0
        for step in range(steps):
            begin = step*batch_size
            end = begin+batch_size
            if step == steps-1: end = len(indices)
            batch = torch.from_numpy(sequencies[indices[begin:end]]).unsqueeze(1).float()
            label_batch = torch.from_numpy(labels[indices[begin:end]]).long()
            out = model(batch)
            loss = criterion(out, label_batch)
            loss.backward()
            optimizer.step()
            corrects += (out.argmax(1) == label_batch).sum().data.numpy()

        acc = corrects/len(labels)
        if acc >0.99:
            break
    return model
            
def CONV_test(model, test, max_seq, results):
    predictions = np.zeros((len(test),12))
    labels = []
    for n, (d,l) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            observation = torch.from_numpy(get_observation(d,t,max_seq).reshape(1,1,-1)).float()
            obs_pred = model(observation).exp().detach().numpy()[0]
            # obs_pred[obs_pred==0] = 1e-6
            # obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)

            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc





def MLP_train(train, params):
    model = MLPClassifier(hidden_layer_sizes = params["hidden_dim"], 
                            learning_rate = "adaptive", 
                            learning_rate_init = params["lr"],
                            max_iter= 10000)
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model



def MLP_test(model, test, max_seq, results):
    predictions = np.zeros((len(test),12))
    labels = []
    for n, (d,l) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            
            observation = get_observation(d,t,max_seq).reshape(1,-1) #d[:t]
            obs_pred = model.predict_proba(observation)[0]
            # obs_pred[obs_pred==0] = 1e-6
            # obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)

            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc




if __name__ == "__main__":
    # data = np.load("./datasets/acticipate/labels_normalized.npy")
    dataset = ActicipateDataset()
    # data = dataset.data[:,[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
    data_raw = dataset.data[:,[0, 1, 2,3,38,39]]
    videos = dataset.videos
    labels = np.array([int(data_raw[b,1]-1) for b,_ in videos])
    train_indices, test_indices = stractfied_data(labels)
   
    train_labels = labels[train_indices]
    train = [(data_raw[b:e,2:],train_labels[i]) for i, (b,e) in enumerate(videos[train_indices])]
    
    test_labels = labels[test_indices]
    test = [(data_raw[b:e,2:],test_labels[i]) for i, (b,e) in enumerate(videos[test_indices])]
   
    for model in ["HMM"]:
        hyperparameters = get_hyperparameters(model)
        hyperparameters["data"] = train
        hyperparameters["model"] = model
        trials = Trials()
        iters = 30 if model == "NB" or model == "HMM" else 200
        best = fmin(fn=bayesian_optimization,
                    space=hyperparameters,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials=trials)
        
        train_clf, test_clf = get_model(model)
        results = []
        params = space_eval(hyperparameters,best)
        final_model = train_clf(train,params )

        del params["data"]
        acc = test_clf(final_model,test,params["max_seq"],results)
        print("{} test acc {:.2f}".format(model, acc*100))
        pickle.dump(trials,open("trials_{}.pkl".format(model),"wb"))
        pickle.dump(params,open("params_{}.pkl".format(model),"wb"))
        pickle.dump({"acc":acc, "results":results},open("results_{}.pkl".format(model),"wb"))

    

