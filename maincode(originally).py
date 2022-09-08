from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
#import os
import glob

#### MTCNN ResNet のモデル読み込み
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#### 画像ファイルから画像の特徴ベクトルを取得(ndarray 512次元)
def feature_vector(image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    feature_vector = resnet(img_cropped.unsqueeze(0))
    feature_vector_np = feature_vector.squeeze().to('cpu').detach().numpy().copy()
    return feature_vector_np

#### 2つのベクトル間のコサイン類似度を取得(cosine_similarity(a, b) = a・b / |a||b|)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
"""
#### 2枚の画像からそれぞれの特徴ベクトルを取得
img1_fv = feature_vector("1.jpg")

img2_fv = feature_vector("2.jpg")

#### 2枚の画像間の類似度を取得
similarity = cosine_similarity(img1_fv, img2_fv)
print(similarity)

"""
img2_fv = feature_vector("11.jpg")

#files = glob.glob('*.jpg')
s=0
for c in range(11,20):
    img1_fv = feature_vector(str(c)+'.jpg')        
    similarity = cosine_similarity(img1_fv, img2_fv)
    
    print(similarity)
    
print("heikinchi",(s-1)/10)
