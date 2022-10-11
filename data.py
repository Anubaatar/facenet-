# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import datetime
# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False (顔の場所を検出して切り取るAIモデル)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() # (512個の数値にするAIモデル)(VGGFace2というデータセットを利用して学習したモデル)  
                                                         # VGGFace2は、約9,000人分の330万枚の画像からなる大規模なデータです。

# Read data from folder

dataset = datasets.ImageFolder('photos') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x): # collate_fn は Dataset から取得した複数のサンプルを結合して、1つのミニバッチを作成する処理を行う関数です。
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True) 
    if face is not None and prob>0.80:
        emb = resnet(face.unsqueeze(0)) 
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx])        

# save data
data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # saving data.pt file
# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) 



def get_datetime():
    dt_now=datetime.datetime.now()
    dt_now_str=str(dt_now.year)+"year "+str(dt_now.month)+"month "+str(dt_now.day)+"day "\
        +str(dt_now.hour)+"hour "+str(dt_now.minute)+"min "+str(dt_now.second)+"sec"
    return dt_now_str

def get_record(data_str):
    dt_now_str=get_datetime()
    enter_str="participated"
    file_str=data_str+","+enter_str+","+dt_now_str
    try:
        with open('kiroku.csv','a',encoding='shift-jis')as f:
            print(file_str,file=f)
    except PermissionError:
        print("ファイルに書き込みできません")

n=0

while True:
    ret, frame = cam.read(0)
    if not ret:
        print("fail to grab frame, try again")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.80:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.80:
                    frame = cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                
                frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)

    cv2.imshow("IMG", frame)
        
    
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
        
    elif k%256==32: # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)
            
        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))
        
    elif k%256==ord('a'): # a to get date time/ to save live pic
        
        cv2.imwrite("participation/pic"+'.'+ str(n)+".jpg",original_frame)
        get_record(name)
        n += 1
        
cam.release()
cv2.destroyAllWindows()