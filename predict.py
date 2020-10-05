#import logging
#logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import numpy as np
import time
import cv2
import os

CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 60
 

def prepare(filepath):
    img_array = cv2.imread(filepath ,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predict(model, filepath):
    test_image = prepare(filepath)
    result = model.predict(test_image.astype('float16'))
    print(CATEGORIES[np.argmax(result)])
    print(result)
    

def validation(model):
    predict(model, 'validation/pod.jpg')
    predict(model, 'validation/dog.jpg')
    predict(model, 'validation/black.jpg')
    predict(model, 'validation/cat1.jpg')
    predict(model, 'validation/potus.jpg')

def register_error(e):
    with open("Err_Log.txt", "a") as file:
        file.write(str(e) + '\n') 
    

def main():
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = tf.keras.models.load_model('model.h5')
        print(model.summary())
        
        try:
            predict(model, 'a.jpg')
        except Exception as e:
            print("Error! Couldn't predict. Please check your image\n")
            register_error(e)
if __name__ == "__main__":
	main()

