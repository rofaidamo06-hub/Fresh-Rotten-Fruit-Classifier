import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree

from sklearn.metrics import accuracy_score



dataset_dir ="Binary_Dataset"
img_size =(64,64)
x =[]
y=[]
classes ={"Fresh":0,
          "Rotten":1,
    }

for category ,label in classes.items():
    folder_path =os.path.join(dataset_dir,category)
    if not os.path.exists(folder_path):
        continue
    for img_name in os.listdir(folder_path):
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_resized=cv2.resize(img,img_size)
        hist = cv2.calcHist([img_resized],[0,1,2],None,[4,4,4],[0,256,0,256,0,256])
        features = cv2.normalize(hist,hist).flatten()
        x.append(features)
        y.append(label)

x=np.array(x)
y=np.array(y)


#scaling or normalization
#x = x/255.0


#training
x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(max_depth=15,random_state=42)
model.fit(x_train,y_train)


# eval.
y_pred = model.predict(x_test)
print(f"Accuracy Score is {accuracy_score(y_test,y_pred)}")

target_names= list(classes.keys())

plt.figure(figsize=(15,10))



plot_tree(model,class_names=target_names,filled=True)
plt.show()

test_img=cv2.imread("Binary_Dataset/Fresh/FreshApple_a_f002.png")
test =cv2.resize(test_img,(64,64))
hist = cv2.calcHist([test],[0,1,2],None,[4,4,4],[0,256,0,256,0,256])
test_features = cv2.normalize(hist,hist).flatten()
pred =target_names[model.predict([test_features])[0]]
print(pred)


import pickle
with open('modelT.pkl', 'wb') as f:
    pickle.dump(model, f)