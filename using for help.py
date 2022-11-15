import tkinter as tk
from tkinter import messagebox
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
import datetime
import time
"""
class Application(tk.Frame):
    g=0
    def __init__(self, master = None):
        super().__init__(master)
        self.match()
        #self.a=a
    
    
    def window(self):
        self.master.title("確認 window")     # ウィンドウタイトル
        self.master.geometry("240x180")       # ウィンドウサイズ(幅x高さ)
        self.a=0    
        names = ['None', 'etuc', 'god', 'ylgu', 'handy', 'アノーシカ', 'taf']
        label1 = tk.Label(text=names[self.g]+'さんですか',font=("MSゴシック", "13", "bold"))
        label1.place(x=40, y=50)
            
            
        # ボタンの作成
         
        button = tk.Button(self.master, text = "はい",command = self.button_end)#.place(x=130, y=100)
        button.place(x=130, y=100)
        
                                           # ボタンの表示名       # クリックされたときに呼ばれるメソッド
            
        #button.pack()
        button1 = tk.Button(self.master, text = "いいえ",command = self.button_end1)#.place(x=50, y=100)
        button1.place(x=50, y=100)
        #self.a=2
           
        #button1.pack()
    def button_end(self):
        self.a=1
        '''クリックされたときに呼ばれるメソッド'''
        print("ボタンがクリックされた hai")
        self.winfo_toplevel().destroy()
        #tk.messagebox(self, 'complete message', 'データ読み込まれました')
        messagebox.showinfo('complete message', 'データ保存された')
    def button_end1(self):
        self.a=2
        '''クリックされたときに呼ばれるメソッド'''
        print("ボタンがクリックされた iie")
        self.winfo_toplevel().destroy()
    def match(self):
        self.window()


def sure():
    root = tk.Tk()
    app = Application(master = root)
    app.mainloop()
    
""""""def test():
    x=Application()
    c=Application
    c.g=5
    if x.a==1:
        print("hai")
    elif x.a==2:
        print("iie")""""""

"""

class mainApp(tk.Frame):
    n=0
    g=0
    def __init__(self, master = None):
        super().__init__(master)
        self.master.title("認証 window")     # ウィンドウタイトル
        self.master.geometry("400x400")       # ウィンドウサイズ(幅x高さ)
        
        button = tk.Button(self.master, text = "顔認証",command = self.match_button,font = ("", 25))#.place(x=130, y=100)
        button.place(x=130, y=100)
    
    
    
    
    
    
    def window(self):
        dlg_modal = tk.Toplevel(self)
        dlg_modal.title("Modal Dialog") # ウィンドウタイトル
        dlg_modal.geometry("250x180")   # ウィンドウサイズ(幅x高さ)

        # モーダルにする設定
        dlg_modal.grab_set()        # モーダルにする
        dlg_modal.focus_set()       # フォーカスを新しいウィンドウをへ移す
        dlg_modal.transient(self.master)   # タスクバーに表示しない

        
        
        self.a=0    
        self.names = ['None', 'etuc', 'god', 'ylgu', 'handy', 'anushka', 'taf']
        
        label1 = tk.Label(dlg_modal, text=self.names[self.g]+'さんですか',font=("MSゴシック", "13", "bold"))
        label1.place(x=40, y=50)
            
            
        # ボタンの作成
         
        button = tk.Button(dlg_modal, text = "はい",command = self.button_end)#.place(x=130, y=100)
        button.place(x=130, y=100)
                                           # ボタンの表示名       # クリックされたときに呼ばれるメソッド  
        #button.pack()
        button1 = tk.Button(dlg_modal, text = "いいえ",command = self.button_end1)#.place(x=50, y=100)
        button1.place(x=50, y=100)
        #self.a=2
        
        #button1.pack()
        # ダイアログが閉じられるまで待つ
        mainApp.wait_window(dlg_modal)       # ウィンドウサイズ(幅x高さ)
    def button_end(self):
        
        '''クリックされたときに呼ばれるメソッド'''
        def get_record(data_str):
            dt_now_str = get_datetime()
            enter_str="participated"
            file_str=data_str+","+enter_str+","+dt_now_str
            try:
                with open('kiroku.csv','a',encoding='shift-jis')as f:
                    print(file_str,file=f)
            except PermissionError:
                print("ファイルに書き込みできません")
    
        def get_datetime():
            dt_now=datetime.datetime.now()
            dt_now_str=str(dt_now.year)+"year "+str(dt_now.month)+"month "+str(dt_now.day)+"day "\
                +str(dt_now.hour)+"hour "+str(dt_now.minute)+"min "+str(dt_now.second)+"sec"
            return dt_now_str
        
        get_record(self.names[self.g])
        
        print("ボタンがクリックされた hai")
        self.destroy()
        #tk.messagebox(self, 'complete message', 'データ読み込まれました')  
        messagebox.showinfo('complete message', 'データ保存された')
    def button_end1(self):
        
        '''クリックされたときに呼ばれるメソッド'''
        print("ボタンがクリックされた iie")
        self.destroy()
        #self.winfo_toplevel().destroy()
        
        
    def match_button(self):
        img_size = 160 #size of resize photos
        #### MTCNN ResNet のモデル読み込み
        mtcnn = MTCNN(image_size=img_size, margin=10)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        names = ['None', 'etuc', 'god', 'ylgu', 'handy', 'shka', 'taf'] #登録済みの名前
        
        def feature_vector(image_path):
            img = Image.open(image_path)
            img_cropped = mtcnn(img)
            feature_vector = resnet(img_cropped.unsqueeze(0))
            feature_vector_np = feature_vector.squeeze().to('cpu').detach().numpy().copy()
            return feature_vector_np
    
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

   
        cam = cv2.VideoCapture(0)
        ### start to save the frame from real time live camera
        while(True):
            ret, frame = cam.read(0)
            cv2.imshow('Face',frame)
            
            k = cv2.waitKey(1)
            if k%256==32: # space to save image
                
                cv2.imwrite("participation/pic"+'.'+ str(self.n)+".jpg",frame)
                self.n+=1
                cam.release()
                cv2.destroyAllWindows()
         ### save the frame from real time live camera until here       
                
                img_cam = Image.fromarray(frame)
                img_cam = img_cam.resize((img_size, img_size))
                img_cam_cropped = mtcnn(img_cam)
                if img_cam_cropped is not None:
                    # if len(img_cam_cropped.size()) != 0:
                    img_embedding = resnet(img_cam_cropped.unsqueeze(0))
                    x2 = img_embedding.squeeze().to('cpu').detach().numpy().copy()
                    #original_frame = frame.copy() # storing copy of frame before drawing on it
                
                
                for c in range(1,7):
                    img1_fv = feature_vector(str(c)+'.jpg')        
                    similarity = cosine_similarity(img1_fv, x2)
                    print(similarity,str(c))
                    
                    if similarity > 0.65:
                        #c = names[c]
                        print("顔認証完成", names[c])
                        self.g=c
                        self.window()
                        
                        
                print("eeey")
                #break
        messagebox.showinfo(self, '顔認証')
        
def sure1():
    root = tk.Tk()
    app = mainApp(master = root)
    app.mainloop()


sure1()






