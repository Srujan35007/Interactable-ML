import time
import csv
import math
import random
from datetime import datetime
print('Imports complete.')


print(f'Is it a Regression model or Classification model ?')
print(f'1. Regression')
print(f'2. Classification')

is_regression = int(input())
if is_regression == 2:
    output_activation = 'softmax'
elif is_regression == 1:
    output_activation = 'relu'
else:
    print('Invalid input. Please select again.')
    is_regression = int(input())

print(f'No. of input perceptrons')
n_input = int(input())
print(f"No. of perceptrons in hidden layers (seperate with a hiphon) '-'")
hidden_layers = (input().split('-'))
print(f'No. of classes (no. of output perceptrons)')
n_outputs = int(input())

if is_regression == 2 and n_outputs > 2:
    loss_fn = 'sparse_categorical_crossentropy'
elif is_regression == 2 and n_outputs == 2:
    loss_fn = 'binary_crossentropy'
elif is_regression == 1:
    loss_fn = 'mse'
else:
    pass


def add_layer(units_):
    units_ = int(units_)
    return f"model.add(layers.Dense(units = {units_}, activation = 'relu'))"


imports = ['import tensorflow as tf\n',
           'from tensorflow import datasets, models, layers\n']
train_and_test = ['X_train = \n',
                  'y_train = \n',
                  'X_test = \n',
                  'y_test = \n']

now = datetime.now()
file_name = now.strftime("%d-%m-%Y-(%H-%M-%S)")
path = f'./tensorflow_ID_{file_name}.py'
with open(path, 'wt') as file:
    file.writelines(f'# Imports \n')
    file.writelines(imports)
    file.writelines('\n')
    file.writelines('# Data \n')
    file.writelines(train_and_test)
    file.writelines('\n # Model \n')
    file.writelines(f'model = models.Sequential()\n')
    file.writelines(
        f'model.add(layers.Flatten(input_shape = [1, {n_input}]))\n')
    file.writelines(
        f"model.add(layers.Dense(units = {n_input}, activation = 'relu'))\n")
    for i in range(len(hidden_layers)):
        file.writelines(add_layer(hidden_layers[i]) + '\n')
    file.writelines(
        f"model.add(layers.Dense(units = {n_outputs}, activation = '{output_activation}'))\n")
    file.writelines(
        f"model.compile(loss = '{loss_fn}', optimizer = 'adam', metrics = ['accuracy'])\n")
    file.writelines(f"print('Model compiled')\n\n")
print(f'Python file generated in the current dir as tensorflow_ID_{file_name}.py')
