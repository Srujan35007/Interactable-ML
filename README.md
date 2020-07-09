# Interactive-ML
Machine learning using command line interface for users who don't want to code.<br>
I would like to thank [Chitla Mani Teja](https://github.com/manitejachitla) for the idea. 
## Key features
- Works offline.
- Automated early stopping (The training process to avoid over fitting).
- Very few third party modules are required.
- Good for beginners as well as intermediate users for machine leanring.

## The functions
### Linear regression
- Excecuted in raw python
- Data preprocessing from the csv, txt, excel files internally.
### Logistic regression
- Excecuted using a small neural network using TensorFlow
- Data preprocessing from the csv, txt, excel files internally.
### Classification
#### From an on device data set
- The model preprocesses the data.
- Creates a suitable neural network in tensorflow.
- Trains and saves the best model.
- Excecutes early stopping as the test loss begins to increase.
#### Gathering data from the webcam
- The user can choose to generate a custom dataset from their webcams.
- The model generates the dataset after the user chooses to stop recording.
- No human intervention needed for dataset creation.
- The model can train on the custom datset if the user mentions to train it.
#### Inbuild datasets in TensorFlow
- The user has a choice to choose from the inbuild datasets in tf.keras
- Automated training and testing.
#### CSV and Excel file datsets
- The program can be trained with not only image data but also numerical data from csv, txt, excel files.
