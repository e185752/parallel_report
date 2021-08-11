""" https://www.codexa.net/opencv_python_introduction/ """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import os
import ray
import time


#画像を表示するための関数を定義
def show_img(Input,Output):
    plt.subplot(121) #画像の位置を指定
    plt.imshow(Input) #画像を表示
    plt.title('Input') #画像の上にInputと表記
    plt.xticks([]) #x軸の目盛りを非表示
    plt.yticks([]) #y軸の目盛りを非表示
    
    plt.subplot(122) #画像の位置を指定
    plt.imshow(Output) #画像を表示
    plt.title('Output') #画像の上にOutputと表記
    plt.xticks([]) #x軸の目盛りを非表示
    plt.yticks([]) #y軸の目盛りを非表示


#一枚の画像ファイル名から読み込み・判断・加工・保存を行う関数
@ray.remote
def making_face(file_name):
    #カスケード型分類器を読み込み
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    face = cv2.imread(file_name)

    faces = face_cascade.detectMultiScale(face, 1.1, 3)

    for (x,y,w,h) in faces:
        face = cv2.rectangle(face,(x,y),(x+w,y+h),(1,1,1),2)
        roi_color = face[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
    try:
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
    except UnboundLocalError:
        pass
    
    filename = file_name.replace('before_face', 'after_face')
    cv2.imwrite(filename,face)
    print(filename)



ray.init(num_cpus=4)
start = time.time()


#保存先のフォルダの生成
for file_name in glob.glob("before_face/*/*/*/*"):
    filename = file_name.replace('before_face', 'after_face')
    os.makedirs(filename, exist_ok=True)
 

#画像１・２を読み込み
file_list = glob.glob("before_face/*/*/*/big/*.jpg", recursive=True)


#演算
#create_picture(file_list)
#create_picture.remote(file_list)
ray.wait([making_face.remote(file_list[num]) for num in range(len(file_list))])
#for num in range(len(file_list)):
#    print(file_list[num])
#    making_face.remote(file_list[num])


#一応時間の計測
finish = time.time()
print("演算時間：", str(finish-start))