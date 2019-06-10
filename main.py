from argparse import ArgumentParser
import keras
from keras import backend as K
import os
from os import walk
import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import tensorflow as tf

size = 128

def load_filenames(data_path):
    filenames = []
    for (dirpath, dirnames, filenames) in walk(data_path.format('x')):
        filenames.extend(filenames)
        break
    training_filenames = []
    validation_filenames = []
    for f in filenames:
        if '.npy' not in f:
            continue
        if 'case1_' in f:
            validation_filenames.append(f)
        else:
            training_filenames.append(f)
    return training_filenames, validation_filenames
    
def train(model_name, data_path, epochs):
    
    data_path = data_path + '/{}/'
    
    training_filenames, validation_filenames = load_filenames(data_path)
    
    if model_name == 'deeplab':
    
        from model3d import Deeplabv3
        deeplab_model = Deeplabv3(
            input_shape=(128, 128, 128, 1),
            classes=1,
            weights=None,
            backbone = 'xception',
            activation='sigmoid'
        )

        from tensorflow.python.keras.optimizers import Adam

        def dice_coef(y_true, y_pred, smooth):
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

        def dice_loss(smooth):
            def dice(y_true, y_pred):
                return 1 - dice_coef(y_true, y_pred, smooth)
            return dice

        def dice_accu(smooth):
            def dice(y_true, y_pred):
                return dice_coef(y_true, y_pred, smooth)
            return dice

        model_dice = dice_loss(smooth=1.)
        dice_score = dice_accu(smooth=1.)

        deeplab_model.compile(
            optimizer = Adam(lr=1e-6),# 1e-4 #lr=7e-4, epsilon=1e-8, decay=1e-6),#1e-2
            loss = model_dice,#'mse',
            metrics=['accuracy', dice_score])

        def windowing(img, lower, upper):

            slope = 1
            intercept = -1024
            img = img * slope + intercept

            img = np.clip(img,lower,upper)
            img = (img - lower) / (upper - lower)

            return img

        def generator(batch_size, filenames):
            while 1:
                shape = (batch_size, size, size, size, 1)

                x_out = np.zeros(shape)
                y_out = np.zeros(shape)

                for i in range(batch_size):
                    filename = random.choice(filenames)
                    x = np.load(data_path.format('x') + filename)
                    y = np.load(data_path.format('y') + filename)

                    x = windowing(x, 100, 1000)# HS window for bone

                    x = np.reshape(x, (size, size, size, -1))
                    y = np.reshape(y, (size, size, size, -1))

                    if random.randint(0,1):# random flip
                        x = np.flip(x, 2)
                        y = np.flip(y, 2)

                    x_out[i] = x
                    y_out[i] = y

                yield (x_out, y_out)

        deeplab_model.fit_generator(  
            generator(1, training_filenames),
            steps_per_epoch=400,
            validation_data=generator(2, validation_filenames),
            validation_steps = 20,
            epochs=epochs)

        save_weights('./weights/model.deeplab.weights.last.h5')
    
    else:
        
        from xception3d import Xception
        model = Xception(
            input_shape = (size, size, size, 1),
            classes = 20
        )
        
        from tensorflow.python.keras.optimizers import Adam
        model.compile(
            optimizer = Adam(lr=1e-2),
            loss = 'categorical_crossentropy',
            metrics=['accuracy'])

        def generator(batch_size, filenames):
            while 1:
                shape = (batch_size, size, size, size, 1)
                shape2 = (batch_size, 20)

                x_out = np.zeros(shape)
                y_out2 = np.zeros(shape2)

                for i in range(batch_size):
                    filename = random.choice(filenames)
                    x = np.load(data_path.format('y') + filename)
                    #print(filename)
                    x = np.reshape(x, (size, size, size, -1))

                    x_out[i] = x

                    label = int(re.sub("\D", "", filename.split('.')[0].split('_')[1]))
                    label_one_hot = [0] * 20
                    label_one_hot[label] = 1

                    y_out2[i] = label_one_hot
                    #print(label_one_hot)

                yield (x_out, y_out2)
                
        model.fit_generator(  
            generator(2, training_filenames),
            steps_per_epoch=100,
            validation_data=generator(4, validation_filenames),
            validation_steps = 10,
            epochs=epochs)

        model.save_weights('./weights/model.xception.weights.last.h5')

def main():
    
    # Command Parser
    
    parser = ArgumentParser()
    
    parser.add_argument("pos1", help="'train' or 'test' or 'cross'")
    parser.add_argument('-m',
                        help='Choose Model ("deeplab" or "xception")',
                        dest='model_name')
    parser.add_argument('-gpu',
                        help='GPU number',
                        dest='gpu_number',
                        default='0')
    parser.add_argument('-data',
                        help='Data path',
                        dest='data_path')
    parser.add_argument('-e',
                        help='Epochs',
                        dest='epochs',
                        default='100')
    
    args = parser.parse_args()
    #print("optional arg:", args)
    
    # GPU setup
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    tf.config.gpu.set_per_process_memory_growth(enabled=True)
    
    # Execution Mode
    
    if args.pos1 == 'train':
        # python3 main.py train -m deeplab -gpu 1 -data ./data/seg_data_original
        train(args.model_name, args.data_path, int(args.epochs))
    elif args.pos1 == 'cross':
            
    


if __name__ == '__main__':
    main() 