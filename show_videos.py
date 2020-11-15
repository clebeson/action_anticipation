
import numpy as np
import cv2
import copy

def draw_gaze(image,gaze):
        canvas = copy.deepcopy(image)
        center = (gaze[0],gaze[1])
        if -1 not in center:
                cv2.circle(canvas,center, 10, (0,255,0), -1)
        return canvas


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
        cap_extern = cv2.VideoCapture("./external.mp4")
        cap_world = cv2.VideoCapture("./world.mp4")
        labels = np.load("action_labels.npy")

        for label in labels:
            ret_ext,frame_ext = cap_extern.read()
            ret_w,frame_w = cap_world.read()
            
            if not (ret_ext and ret_w): break

            frame_w = draw_gaze(frame_w,label[-2:])
            frame_ext = draw_class(frame, label[1])
            frame_concat = np.concatenate((framew,frame),axis=1)
            cv2.imshow("Action Dataset",frame_concat)
            
            if cv2.waitKey(25) == 27:return #ESC



if __name__ == "__main__":
    show_videos()