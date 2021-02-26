# DeepLearningProject1
In this project we tested a CNN and and ensemble of MLPs on the classical MNIST dataset.
The follwing code is included in the different files:
  - functions: training, evaluating and the function that combines the output from the MLPs
  - linear_nets: The implementation of the different MLPs
  - CNNs: implementation of LeNet5 and an a deeper version of LeNet5
  - plots_and_stuff: function to plot train and test performance
  - fileIO: reading in the data
  - launch: here all functions are called and the go function is implemented to run the project from google colab

To run this project on Google colab simply execute the following code on colab: \
!git clone https://github.com/JanRob23/DeepLearningProject1.git \
from google.colab import drive\
drive.mount("/content/drive")\
(change to where MNIST is located in your drive)\
train = '/content/drive/MyDrive/data/mnist_train.csv' \
test = '/content/drive/MyDrive/data/mnist_test.csv'\
%cd DeepLearningProject1\
from launch import go\
go(train, test)\

Disclaimer: Due to having progress bars for notebooks (google colab) this code might give an error when run without using notebooks
  
