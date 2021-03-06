import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import csv
from tensorflow.keras import models, layers
import random
import time
import cv2
import math
import pickle
print('Imports complete.')


def k_nearest_neighbours(train_data, unknown_point,  n=1):
    def get_distance(point_unknown, point_target):
        if len(point_target) != len(point_target):
            print('Dimesions of given data point do not match the train data')
            k_means(train_data)
        else:
            distance = 0
            for i in range(len(point_target)):
                distance += (point_target[i] - point_unknown[i])**2
            return math.sqrt(distance)
    def get_majority(array):
        all_labels_set = list(set([element[1] for element in array]))
        all_labels = [element[1] for element in array]
        max_ = 0
        for label in all_labels_set:
            if max_ < all_labels.count(label):
                max_ = all_labels.count(label)
                max_label = label
        return max_label
    distances = []
    for data in train_data:
        X_point, label = data
        distances.append((get_distance(unknown_point, X_point), label))
    sort_criteria = lambda x : x[0]
    distances.sort(key=sort_criteria)
    distances = distances[:n]
    return get_majority(distances)


class linear_regression():
    def __init__(self):
        pass

    def fit(self, data_set):
        # returns m and b
        sigma_x = 0
        sigma_y = 0
        sigma_x_squared = 0
        sigma_xy = 0
        n = len(data_set)
        for point in data_set:
            x,y = point
            sigma_x += x
            sigma_y += y
            sigma_x_squared += x**2
            sigma_xy += x*y
        
        m = (n*sigma_xy - (sigma_x*sigma_y))/(n*sigma_x_squared - sigma_x**2)
        b = (sigma_y*sigma_x_squared - sigma_x*sigma_xy)/(n*sigma_x_squared - sigma_x**2)

        self.m = m
        self.b = b
    
    def get_params(self):
        # returns m and b
        return (self.m, self.b)

    def test(self, test_point_x):
        # returns the y coordinate of the wanted x coordinate
        return self.m*(test_point_x) + self.b


def files_without_extention(file_name):
    # Seperate filename from extension
    temp = ''
    for i in range(len(fileName)-1):
        if fileName[i] == '.':
            break
        else:
            temp += fileName[i]
    return temp


def convert_to_csv(path_to_file, file_name):
    # Convert Text, Excel files into CSV file
    total_path = path_to_file + '\\' + file_name
    file_name_without_extention = files_without_extention(file_name)
    renamed_path = path_to_file + '\\' + file_name_without_extention + '.csv'
    if file_name.endswith('.txt') or file_name.endswith('.csv') or file_name.endswith('.xlsx'):
        os.rename(total_path, renamed_path)
    else:
        print('Invalid extension. Try .txt .csv .xlsx files.')


def split_data(normalized_data):
    # Split data into train data and test data
    len_data = len(normalized_data)
    train_data = normalized_data[:(int(len_data*0.83))]
    test_data = normalized_data[(int(len_data*0.83)):]
    del(normalized_data)
    return (train_data, test_data)


def normalize_dataset(data_set):
    # For normalizing the data from a CSV file.(For classification only)
    print(f'Normalizing the dataset.')
    max_values_for_each_column = [0]*(len(data_set[0])-1)
    for i in range(len(data_set)):
        for j in range(len(max_values_for_each_column)):
            if max_values_for_each_column[j] < data_set[i][j]:
                max_values_for_each_column[j] = data_set[i][j]
            else:
                pass
    for i in range(len(data_set)):
        for j in range(len(max_values_for_each_column)):
            data_set[i][j] = data_set[i][j]/max_values_for_each_column[j]
    print('Normalizing the dataset completed.')
    return data_set


def one_hot(num, n_outputs):
    # Make classes into One-hot vector
    one_h = []
    for i in range(n_outputs):
        if i != num:
            one_h.append(0)
        else:
            one_h.append(1)
    return one_h


def get_training_data_from_cam():
    # Get data from Webcam
    wait_time = 8 # seconds
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        now = time.time()
        elapsed = (now - start_time)//1
        #print(elapsed)
        if elapsed < wait_time + 1:
            text = f'Recording starts in {wait_time - elapsed + 1}'
        else:
            text = f'Rec : {count} Frames'
            filename = f'C:\\Users\\S.RAMACHANDRAM\\Desktop\\New\\a{count}.jpg'
            cv2.imwrite(filename, frame)
            count += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (0,100), font, 1, (100,250,0), thickness=1)
        cv2.imshow('Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def test_with_webcam(trained_model, input_shape, classes_list):
    cap = cv2.VideoCapture(0)
    start = time.time()
    class_ = ''
    confidence = 0
    while True:
        ret, frame = cap.read()
        elapsed = round((time.time() - start),1)
        reshaped = np.asarray(cv2.resize(frame, input_shape)) # Tweaking Needed
        out = trained_model.predict([reshaped]))[0]
        class_ = classes_list[np.argmax(out)]
        confidence = f'Confidence : {str(max(out)*100)}' + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, class_, (0,16), font, 0.5, (100,250,0), thickness=1)
        cv2.putText(frame, confidence, (0,36), font, 0.5, (100,250,0), thickness=1)
        cv2.imshow('Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_data_from_path(path_, folder_name_or_file_name, is_regression = False):
    # Make dataset ready to use from Data folders or Data file.
    path_string = '\\'.join(path_.split('\\'))
    path_to_data_file = path_string + '\\' + folder_name_or_file_name
    if is_regression is False:
        num_classes = 0
        classes = []
        for roots, dirs, files in os.walk(path_to_data_file):
            if dirs != []:
                classes.append(dirs)
            else:
                pass
        classes = classes[0]
        num_classes = len(classes)
        data = []
        for i in range(len(classes)):
            files = []
            path_to_class = path_to_data_file + classes[i]
            for filename in os.listdir(path_to_class):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    files.append(path_to_class + filename)
                else:
                    pass
            for file in files:
                img = cv2.imread(file)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append([np.array(img_rgb)/255, one_hot(i, len(classes))])
        return data
    elif is_regression:
        # Yet to be done
        pass


def get_inputs():
    # Get user inputs from the user.
    print(f'Select a task:')
    print(f'1. Classification.')
    print(f'2. Regression.')
    print(f'3. Load trained models.')
    class_ = int(input())
    if class_ == 1:
        print(f'Source of training data:')
        print(f'1. Path to the data-set.')
        print(f'2. Make training data through webcam.')
        source_ = int(input())
        if source_ == 1:
            print(f'Paste the path to you data-set.')
            path_ = input()
            print(f'The name of the folder that you stored all the classes')
            folder = input()
            data = get_data_from_path(path_, folder, is_regression=False)
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



# The lines from now on are for testing.
'''
net = make_image_classification_model((80, 80, 3), 3)
img = np.tile(0.5, (80, 80, 3)).reshape(1, 80, 80, 3)  # Demo image for testing
print(np.argmax(net.predict(img)[0]))
'''

# Making the regression model
net = make_regression_model((1,1)) # Making the regression model
#print(net.summary())

# Demo input for regression(Logistic)
arr = [[i*5, i] for i in range(10000)]
print(arr[1])
elem = [arr[i][0] for i in range(10000)]
label = [arr[i][1] for i in range(10000)]

# training the regression model
train_model(net, arr, 1, 'test1', True)

new_model = models.load_model(f'./Ignore/test1')
val_loss, val_acc = new_model.evaluate(elem, label)
print(val_acc*100, val_loss)
