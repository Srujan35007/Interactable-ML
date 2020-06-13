import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import csv
from tensorflow.keras import models, layers
import random
import time
print('Imports complete.')


def get_training_data_from_cam():
    pass

def get_data_from_path(path_):
    pass



def get_inputs:
    print(f'Select a task:')
    print(f'1. Classification.')
    print(f'2. Regression.')
    print(f'3. Load trained models.')
    class_ = input()
    if class_ == 1:
        print(f'Source of training data:')
        print(f'1. Path to the data-set.')
        print(f'2. Make training data through webcam.')
        source_ = input()
        if source_ == 1:
            print(f'Paste the path to you data-set.')
            path_ = input()
            data = get_data_from_path(path_)
        elif source_ == 2:
            print(f'WebCam opening shortly.')
            


    elif class_ == 2:
        pass
    elif class_ == 3:
        pass
    else:
        print('Input invalid.')
        get_inputs()
    


def make_image_classification_model(input_shape_, no_of_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(1, 3, (1, 1),
                            activation='relu', input_shape=input_shape_))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(32, 4, (1, 1), activation='relu'))
    model.add(layers.MaxPool2D((4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(no_of_classes, activation='softmax'))
    return model
    print('Image classification model Created.')


def make_regression_model(no_of_input_parameters):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=no_of_input_parameters))
    model.add(layers.Dense(units=no_of_input_parameters[1], activation='relu'))
    for _ in range(2):
        model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation=None))
    return model
    print('Regression model created.')


def train_model(model, data, no_of_classes, checkpoint_file_name, is_regression=False):
    random.shuffle(data)
    train_data_temp = data[:(int(len(data)*0.83))]
    test_data_temp = data[(int(len(data)*0.83)+1):]
    print(len(test_data_temp))
    train_data_elements = []
    train_data_labels = []
    test_data_elements = []
    test_data_labels = []
    for i in range(len(train_data_temp)):
        train_data_elements.append(train_data_temp[i][0])
        train_data_labels.append(train_data_temp[i][1])
    del(train_data_temp)
    for i in range(len(test_data_temp)):
        test_data_elements.append(test_data_temp[i][0])
        test_data_labels.append(test_data_temp[i][1])
    del(test_data_temp)
    if is_regression:
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        PATIENCE = 10
        BATCH_SIZE = int(len(train_data_elements)/100)
        VERBOSE = False
        SAVE_MODEL = True
        checkpoint_filename = checkpoint_file_name

        val_loss_list = []
        val_acc_list = []
        epoch_count = 1
        train_flag = True
        patience_temp = PATIENCE
        X_train, y_train = train_data_elements, train_data_labels
        X_test, y_test = test_data_elements, test_data_labels
        del(train_data_elements)
        del(train_data_labels)
        del(test_data_elements)
        del(test_data_labels)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) != 0:
            device = tf.device('GPU:0')
            print('Running on the GPU.')
        else:
            device = tf.device('CPU')
            print('Running on the CPU.')

        with device:
            while train_flag:
                model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                        epochs=1, verbose=VERBOSE)
                val_loss, val_acc = model.evaluate(X_test, y_test, verbose=VERBOSE, batch_size = 1)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc*100)
                # train_optimizer part
                if len(val_loss_list) == 1:
                    k = -1
                else:
                    k = -2
                if val_loss != min(val_loss_list) or val_loss == val_loss_list[k]:
                    patience_temp -= 1
                else:
                    patience_temp = PATIENCE
                    if SAVE_MODEL is True:
                        if Path(f'./{checkpoint_filename}').is_file():
                            os.remove(f'./Ignore/{checkpoint_filename}-regression')
                        model.save(f'./Ignore/{checkpoint_filename}-regression', overwrite = True)
                        if VERBOSE is True:
                            print(f'New model saved as <{checkpoint_filename}-regression>.')
                    else:
                        pass
                    
                if patience_temp == 0:
                    train_flag = False
                else:
                    train_flag = True
                epoch_count += 1
    
    else:
        if no_of_classes == 2:
            loss_fn = 'binary_crossentropy'
        elif no_of_classes > 2:
            loss_fn = 'sparse_categorical_crossentropy'
        model.compile(optimizer = 'adam', loss = loss_fn, metrics = ['accuracy'])

        PATIENCE = 6
        BATCH_SIZE = int(len(train_data_elements)/100)
        VERBOSE = False
        SAVE_MODEL = True
        checkpoint_filename = checkpoint_file_name

        val_loss_list = []
        val_acc_list = []
        epoch_count = 1
        train_flag = True
        patience_temp = PATIENCE
        X_train, y_train = train_data_elements, train_data_labels
        X_test, y_test = test_data_elements, test_data_labels
        del(train_data_elements)
        del(train_data_labels)
        del(test_data_elements)
        del(test_data_labels)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) != 0:
            device = tf.device('GPU:0')
            print('Running on the GPU.')
        else:
            device = tf.device('CPU')
            print('Running on the CPU.')

        with device:
            while train_flag:
                model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                        epochs=1, verbose=VERBOSE)
                val_loss, val_acc = model.evaluate(X_test, y_test, verbose=VERBOSE, batch_size = 1)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc*100)
                # train_optimizer part
                if len(val_loss_list) == 1:
                    k = -1
                else:
                    k = -2
                if val_loss != min(val_loss_list) or val_loss == val_loss_list[k]:
                    patience_temp -= 1
                else:
                    patience_temp = PATIENCE
                    if SAVE_MODEL is True:
                        if Path(f'./{checkpoint_filename}').is_file():
                            os.remove(f'./Ignore/{checkpoint_filename}-classification')
                        model.save(f'./Ignore/{checkpoint_filename}-classification', overwrite = True)
                        if VERBOSE is True:
                            print(f'New model saved as <{checkpoint_filename}-classification>.')
                    else:
                        pass
                    
                if patience_temp == 0:
                    train_flag = False
                else:
                    train_flag = True
                epoch_count += 1
        print(f'New model saved as <{checkpoint_filename}-classification>.')
    
'''
net = make_image_classification_model((80, 80, 3), 3)
img = np.tile(0.5, (80, 80, 3)).reshape(1, 80, 80, 3)  # Demo image for testing
print(np.argmax(net.predict(img)[0]))
'''


net = make_regression_model((1,1))
#print(net.summary())

arr = [[i*5, i] for i in range(10000)]
print(arr[1])
elem = [arr[i][0] for i in range(10000)]
label = [arr[i][1] for i in range(10000)]


train_model(net, arr, 1, 'test1', True)

new_model = models.load_model(f'./Ignore/test1')
val_loss, val_acc = new_model.evaluate(elem, label)
print(val_acc*100, val_loss)
