#!/usr/bin/python
from __future__ import print_function
import sys
sys.path.insert(0, 'pytorch-openpose/python')
sys.path.insert(0, 'pytorch-openpose/')
import numpy as np 
import scipy.ndimage as ndi 
import argparse
import os 
import csv
import cv2
import matplotlib
from timeit import default_timer as timer
import pims
from functools import reduce
from moviepy.editor import *
import scipy.misc as m
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from  scipy.spatial.distance import cosine
import numpy as np 
import moviepy as moviepy
from moviepy.editor import *
import pandas as pd
import copy
import math
from scipy import signal
from scipy import stats


import imageio
imageio.plugins.ffmpeg.download()

def draw_bodypose(image, joints):
        canvas = copy.deepcopy(image)
        stickwidth = 4

        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
                
        cur_canvas = canvas.copy()
        for i in range(17): 
                if i in [6,9,12,13,14,15,16]: continue 
                values = joints[np.array(limbSeq[i]) - 1]
                if -1 in values:continue
                Y = values[:,0]
                X = values[:,1]

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        return canvas




def draw_head(image, joints):
        canvas = copy.deepcopy(image)
        stickwidth = 4

        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
                
        cur_canvas = canvas.copy()
        for i in [12,13,14,15,16]:    
                values = joints[np.array(limbSeq[i]) - 1]
                Y = values[:,0]
                X = values[:,1]

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        return canvas


def draw_ballpose(image,ball):
        canvas = copy.deepcopy(image)
        center = (ball[0],ball[1])
        if -1 not in center:
                cv2.circle(canvas,center, 15, (0,0,255), -1)
        return canvas

def draw_gaze(image,gaze):
        canvas = copy.deepcopy(image)
        center = (gaze[0],gaze[1])
        if -1 not in center:
                cv2.circle(canvas,center, 10, (0,255,0), -1)
        return canvas

def draw_hand_center(image,hands):
        canvas = copy.deepcopy(image)
        for hand in hands:
                center = (hand[0],hand[1])
                if -1 not in center:
                        cv2.circle(canvas, center, 10, (255, 0, 0), thickness=-1)
        return canvas
def draw_handpose(image, hands, show_number=False):
    canvas = copy.deepcopy(image)
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for hand in hands:
        # for ie, e in enumerate(edges):
        #         if np.sum(np.all(hand[e], axis=1)==0)==0:
        #                 x1, y1 = hand[e[0]]
        #                 x2, y2 = hand[e[1]]
        #                 cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])*255, thickness=2)

        # for i, (x,y) in enumerate(hand[:-2]):
        #         cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        #         if show_number:
        #                 cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
        for i in range(len(hand)):
                x,y = (hand[-2]+0.6*hand[-1]).astype(int)
                cv2.circle(canvas, (x, y), 10, (255, 0, 0), thickness=-1)
        # hand = hand[:-2]
        # hand[hand<0]=0
        # total = (hand>0).sum(0)+1e-18

        # x,y = (hand.sum(0)/total).astype(int)

        # cv2.circle(canvas, (x, y), 10, (0, 255, 0), thickness=-1)
    return canvas



def get_joints(candidate,subset):   
    joints = np.ones((18,2))*-1
    joints_number = range(18)
    for n in range(len(subset)):
        ind = subset[n,[1,2,5]].astype(int)
        if -1 not in ind:
            values = candidate[ind,0:2]
            x,y = values.mean(0)
            if x>200 and x<400 and y>120:
                indices = subset[n,joints_number].astype(int)
                for i, index in enumerate(indices):
                        if i==-1:continue
                        joints[i]= candidate[index,0:2]
                break

    return joints

 

def mouse_event(event,x,y,flags,param):
    global mx, my
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x,y



def generate_body_labels():
        body_estimation = Body('pytorch-openpose/model/body_pose_model.pth')
        hand_estimation = Hand('pytorch-openpose/model/hand_pose_model.pth')
        def get_body(img):
                candidate, subset = body_estimation(img)
                candidate = get_joints(candidate,subset)
                subset = np.array([[i if candidate[i,0] >-1 else -1 for i in range(18)]])
                return candidate
        
        def get_hand(img,body):
                subset = np.array([[i if body[i,0] >-1 else -1 for i in range(18)]])
                hands_list = util.handDetect(body, subset, img)
                hand_joints= np.ones((2,23,2))*-1
                for x, y, w, is_left in hands_list:
                        hand = 0 if is_left else 1
                        peaks = hand_estimation(img[y:y+w, x:x+w, :])
                        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                        hand_joints[hand,:-2]= peaks
                        hand_joints[hand,-2:]= np.array([[x,y],[w,w]])
                return hand_joints

        files = glob.glob("out.mp4")
        #label = pd.read_csv("labels2.csv")
        hand_size = 2*23*2
        skl = 18*2
        print("total:",len(files))
        for file in files:
                print(file)
                video = []
                i = 0
                start = timer()
                images = pims.Video(file)
                total = len(images)
                videos = np.ones((total,1+hand_size+skl))
                np.save("./labels_cut.npy",videos)
                for frame,img in enumerate(images):
                        # ball = label[(label.video==file.split("/")[-1]) & (label.frame == frame)].values
                        # ball = ball[:,-2:].reshape(-1)
                        #img = img[:,:,[2,0,1]]
                        body= get_body(img)
                        hand = get_hand(img,body)
                        videos[frame,0]=frame
                        videos[frame,1:skl+1]=body.flatten()
                        videos[frame,skl+1:]=hand.flatten()
                        if frame % 500 == 0:
                                print("{}/{} time/frame = {}".format(frame,total, (timer()-start)/(frame+1)))


        np.save("./labels_cut.npy",videos)

def check_skeleton(joints):
        x,y = joints[[1,2,5]].mean(0)
        if x>200 and x<400 and y>120:return True    
        return False

def check_hand(hands):
        for hand in hands:
                values = hand[:-2]>0
                if values.sum() < 5: return False
                return True


def recalc_skeleton():
        body_estimation = Body('pytorch-openpose/model/body_pose_model.pth')
        hand_estimation = Hand('pytorch-openpose/model/hand_pose_model.pth')
        def get_body(img):
                candidate, subset = body_estimation(img)
                candidate = get_joints(candidate,subset)
                subset = np.array([[i if candidate[i,0] >-1 else -1 for i in range(18)]])
                return candidate
        
        def get_hand(img,body):
                subset = np.array([[i if body[i,0] >-1 else -1 for i in range(18)]])
                hands_list = util.handDetect(body, subset, img)
                hand_joints= np.ones((2,23,2))*-1
                for x, y, w, is_left in hands_list:
                        hand = 0 if is_left else 1
                        peaks = hand_estimation(img[y:y+w, x:x+w, :])
                        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                        hand_joints[hand,:-2]= peaks
                        hand_joints[hand,-2:]= np.array([[x,y],[w,w]])
                return hand_joints
                
    
        dir = "videos/labels/"
        labels_dict = pickle.load(open("label_complete.pikle", 'rb'))
        remaining = 240
        for name, video in labels_dict.iteritems():
                start = timer()
                total_images = 1
                images =  pims.Video(dir+name)
                for label in video:
                        recalc = False
                        body = label["body"]
                        hand= np.ones((2,23,2))
                        hand[:,:-2] = label["hand"]
                        frame = label["frame"]
                        image = images[frame]
                        if not check_skeleton(body): 
                                recalc = True
                                label["body"] = get_body(image)
                        
                        subset = np.array([[i if body[i,0] >-1 else -1 for i in range(18)]])        
                        box = util.handDetect(body, subset, image)
                        
                        for x, y, w, is_left in box:
                                rl = 0 if is_left else 1
                                hand[rl,-2:]= np.array([[x,y],[w,w]])
                        
                        if not check_hand(hand):hand = get_hand(image,label["body"]); recalc = True
              
                        label["hand"] =hand

                        if recalc: total_images+=1
                remaining -=1
                        
                print("total images = {}/{},  remaining = {}, time/frame = {},".format(total_images,len(images), remaining, (timer()-start)/i))
        pickle.dump(labels_dict, open( "label_complete_recalc.pikle", "wb" ), protocol=2)


def draw_class(image, label):
        actions = {
                1:"Place Left",
                2:"Give Left",
                3:"Place Middle",
                4:"Give Middle",
                5:"Place Right",
                6:"Give Right",
                7:" Pick Left",
                8:" Receive Left",
                9:" Pick Middle",
                10:"Receive Middle",
                11:"Pick Right",
                12:"Receive Right"
        }
        canvas = copy.deepcopy(image)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (380,50)
        fontScale              = 0.7
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(canvas,"{} ({})".format(actions[label],label), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

        return canvas


def show_videos():
        files = glob.glob("labels/*.mp4")
        labels_dict = pickle.load(open("label_complete.pikle", 'rb'),encoding = 'latin1')
        draw_hand = True
        draw_body = True
        draw_ball = True
        empty = False
        empty_image = np.zeros((480,640,3))
        cv2.imshow("Action Dataset",empty_image)
        for file in files:
                name = file.split("/")[-1]
                video_labels = labels_dict[name]
                frames = pims.Video(file)
                
                for i,frame_label in enumerate(video_labels):
                        frame = empty_image if empty else frames[i]
                        frame = draw_bodypose(frame,frame_label["body"].astype(int)) if draw_body else frame
                        frame = draw_handpose(frame,frame_label["hand"].astype(int)) if draw_hand else frame
                        frame = draw_ballpose(frame,frame_label["ball"].astype(int)) if draw_ball else frame
                        
                        cv2.imshow("Action Dataset",frame[:,:,[2,1,0]])
                        k = cv2.waitKey(10)
                        if k == 27:return
                        if k == 98:draw_ball = not draw_ball #B
                        if k == 101:empty = not empty #E
                        if k == 104:draw_hand = not draw_hand #H
                        if k == 106:draw_body = not draw_body #J
                        if k == 32: 
                                while cv2.waitKey(30) != 32:continue
                                
                        
def show_video():
        #labels_body = np.load("labels_body.npy").astype(int)
        draw_hand = True
        draw_body = True
        draw_ball = True
        empty = False
        empty_image = np.zeros((480,640,3))
        frames = pims.Video("./dataset/external.mp4")
        framesw = pims.Video("./dataset/world.mp4")
        labels = np.load("labels_complete_cut.npy").astype(int)
        #np.save("./dataset/action_labels.npy",labels[:,[0,1,-2,-1]])
        
        # labels = np.load("labels_complete.npy").astype(int)
        # fixation = np.load("./dataset/fixation_cut.npy")
        # x = labels[:,2:]
        
        # x1 = pd.DataFrame(x).rolling(3,1,True).mean().values
        # r = pd.DataFrame(x1).rolling(10,1,True)
        # m = r.mean().values
        # s = r.std(ddof=0).values
        # x1 = np.where(np.abs((x1-m)/s) <  1., x1, m)
        # x1[:,0:2] =  np.where(x[:,0:2] <= 0, 0, x1[:,0:2])
        # x1 = np.where(x1 == 0, -1, x1)
        # j= np.array([18,19,20,21,24,25,26,27])+2
        # x1[:,j] = -1
        # labels[:,2:] = x1
        # hand = labels[:,-23*4:].reshape((-1,2,46))
        # hand = hand[:,:,-4:].reshape((-1,8))
        # print(hand.shape)
        # labels = labels[:,:-(23*4-6)]
        # labels[:,40:42] = hand[:,4:6]+0.6*hand[:,6:8]    
        # labels[:,42:44] =   hand[:,0:2] + 0.6*hand[:,2:4] 
         
        # fixation[:,0] = ((fixation[:,0] / 800.) * 640).astype(int)
        # fixation[:,1] = ((fixation[:,1] / 600.) * 480).astype(int)
        # labels[:,44:] = fixation[:len(labels)]

        # np.save("labels_complete_cut.npy",labels.astype(int))
        # return
        
        cv2.imshow("Action Dataset",empty_image)
        labels = labels[439:540]
        for i, label in enumerate(labels):
                print(i)
                frame = frames[label[0]]
                framew = framesw[label[0]]
                #print(label.shape)
                gaze = label[-2:]
                ball = label[2:4]
                body = label[4:40].reshape((18,2)).astype(int)
                hand = label[40:44].reshape((2,2)).astype(int)
                frame = empty_image if empty else frame
                frame = draw_bodypose(frame,body) if draw_body else frame
                frame = draw_hand_center(frame,hand) if draw_hand else frame
                frame = draw_ballpose(frame,ball.astype(int)) if draw_ball else frame
                framew = draw_gaze(framew,gaze)
                frame = draw_class(frame, label[1])
                frame = np.concatenate((framew,frame),axis=1)
                cv2.imshow("Action Dataset",frame[:,:,[2,1,0]])
                k = cv2.waitKey(500)
                if k == 27:return
                if k == 98:draw_ball = not draw_ball #B
                if k == 101:empty = not empty #E
                if k == 104:draw_hand = not draw_hand #H
                if k == 106:draw_body = not draw_body #J
                if k == 32: 
                        while cv2.waitKey(30) != 32:continue




if __name__ == "__main__":
        # import util
        # import model
        # from hand import Hand
        # from body import Body
        # generate_body_labels()
        # recalc_skeleton()
        
        show_video()
        #show_videos()




