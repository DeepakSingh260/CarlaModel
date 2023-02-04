import tensorflow.compat.v2 as tf
import cv2
import glob
import numpy as np
path = "SteerValues/steer_values.txt"
img_folder = "Images/*.png"
imgs = []
index = 0 
for img in glob.glob(img_folder):
    print(index,img)
    imgs.append(cv2.imread(img)[110:220,:,:1])
    index+=1
    if index==40000:break
# imgs.sort()
imgs = imgs[:40000]
labels = []
index = 0 
file1 = open(path ,"r")
while True:
   val = file1.readline()
   if not val or index == 40000:
       break
   val = float(val)*100
   val = tf.cast(val,tf.float32)
   labels.append(val)
   index+=1
 


# Dataset =  tf.data.Dataset.from_tensor_slices( imgs , labels)

Model = tf.keras.Sequential([
    tf.keras.Input(shape = (110, 220, 1)),
    tf.keras.layers.Conv2D(24 ,(5,5) , strides = (2,2) , padding ="same"),
    tf.keras.layers.Conv2D(36 ,(5,5) , strides = (2,2) , padding ="same"),
    tf.keras.layers.Conv2D(48 ,(5,5) , strides = (2,2) , padding ="same"),
    tf.keras.layers.Conv2D(64 ,(3,3)  , padding ="same"),
    tf.keras.layers.Conv2D(64 ,(3,3) , padding ="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000 ,activation = "elu"),
    tf.keras.layers.Dense(100 ,activation = "elu"),
    tf.keras.layers.Dense(10 ,activation = "elu"),
    tf.keras.layers.Dense(1)
])
opt = tf.keras.optimizers.Adam(
   learning_rate = 1e-5
)

Model.compile(loss = "mean_squared_error" , optimizer= opt, metrics = ['accuracy'])
Model.fit(np.asarray(imgs), np.asarray(labels),epochs = 5, verbose = 1)

Model.save("steer_model")
file1.close()

