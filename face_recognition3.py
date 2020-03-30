# -*- coding: utf-8 -*-
# @Author：sxw  Time:2020/3/1
import cv2
import os
import numpy as np
import torch as t
from torch import nn

face_cascade = cv2.CascadeClassifier(
    r'D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    r'D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
IMAGE_SIZE = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lin1 = nn.Linear(128 * (IMAGE_SIZE // 8) ** 2, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 3)

    def forward(self, x):
        cov1_x = self.cov1(x)
        cov2_x = self.cov2(cov1_x)
        cov3_x = self.cov3(cov2_x)
        reshape_x = cov3_x.view(cov3_x.shape[0], -1)
        lin1_x = self.lin1(reshape_x)
        lin2_x = self.lin2(lin1_x)
        output_x = self.lin3(lin2_x)
        return output_x


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / sum(exp_x)
    return softmax_x


def predict_image_from_video():
    cnn_model = t.load('cnn_model.pkl')
    cap = cv2.VideoCapture(0)

    while True:
        status, img = cap.read()
        lable = 0
        if not status:
            print('摄像头异常，强制推出！！')
            break
        img_w, img_h, _ = img.shape
        img_vet = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_loc = face_cascade.detectMultiScale(img_vet, scaleFactor=1.2, minSize=(100, 100))
        if len(faces_loc) != 0:
            x, y, w, h = faces_loc[0]
            if w > 150:
                x_ = 0 if (x - 10 < 0) else x - 10
                y_ = 0 if (y - 10 < 0) else y - 10
                image_array = img[y_:y + h + 10, x_:x + w + 10, :]
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                resize_image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
                resize_image_array = resize_image_array.astype('float32')
                resize_image_array /= 255
                tensor_x = t.from_numpy(resize_image_array).unsqueeze(0).unsqueeze(0)
                predict_y_pro = cnn_model(tensor_x)
                predict_y = t.argmax(predict_y_pro)
                lable = predict_y.item()
                predict_y_pro = max(softmax(predict_y_pro.squeeze().data.numpy()))
                if lable == 0:
                    lable_text = 'other'
                elif lable == 1:
                    lable_text = 'songxiwen'
                else:
                    lable_text = 'baobao'
                cv2.putText(img, lable_text+' p:{:.2f}%'.format(predict_y_pro*100),
                            (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(128, 125, 0), thickness=2)
        cv2.imshow('video', img)
        wait_time = 100 if (lable in [1, 2]) else 1
        k = cv2.waitKey(wait_time)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_image_from_video()
