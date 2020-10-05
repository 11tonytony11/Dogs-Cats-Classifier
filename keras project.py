import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import numpy as np
import pickle
import random
import cv2
import os

CATEGORIES = ["Cat", "Dog"]
DATADIR = "D:\\AI\\Datasets\\PetImages\\"
IMG_SIZE = 60


"""
This function creates nerual network DNN (Type CNN)
Input: Images - to get tensor shape for input conv layer
Output: DNN Sequential Model
"""
def create_nerual_network(imgs):
        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=imgs.shape[1:]))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        model.add(tf.keras.layers.Conv2D(64, (2, 2)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (2, 2)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (2, 2)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (2, 2)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))

        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dropout(0.4))
                
        model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
    
        return model
    
"""
This function scans folder, then loads all images and labes. it also does data preperation
Input: None - uses const for dir
Output: np array for images and no array for labels
"""
def prepare_data_for_training():
        training_images = []
        training_labels = []
        training_data = []
    
        for category in CATEGORIES:

                path = os.path.join(DATADIR,category)   
                class_num = CATEGORIES.index(category)

                for img in os.listdir(path):
                        try:
                                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                                training_data.append([new_array, class_num])
                        except Exception as e: 
                                print("Image error at: ", os.path.join(path,img))
                                with open("Err_Log.txt", "a") as file:
                                         file.write(os.path.join(path,img) + '\n')                       
        random.shuffle(training_data)
    
        for features,label in training_data:
                
                training_images.append(features)
                training_labels.append(label)


        #Convert to numpy tensor, save Data-set and return
        save_dataset(np.array(training_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(training_labels))
        return np.array(training_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(training_labels)
    
def load_dataset():
        pickle_in = open("X.pickle","rb") #images
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb") #labes
        y = pickle.load(pickle_in)

        return X, y #X and y are common names for images and labels...

def save_dataset(X, y):
    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()




def main():
        train_images, train_labels = load_dataset()
        train_images = train_images.astype('float16')
        train_images = train_images / 256.

        model = create_nerual_network(train_images)
        #model = tf.keras.models.load_model('model2.h5')
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("Training: ")
        
        model.fit(train_images, train_labels, epochs = 10, verbose=2)
        
        x = input("Confirm Saving: ")
        model.save("model1.h5")
        
        print("Done!\nModel Saved!")
        x = input()
        return 0
        
    
if __name__ == "__main__":
    main()

#used only for the first run...
#model = create_nerual_network(train_images)
#model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
