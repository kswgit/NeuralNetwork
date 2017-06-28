#!/bin/bash

mkdir mnist_data
cd mnist_data
wget https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv
wget https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv
wget http://www.pjreddie.com/media/files/mnist_train.csv
wget http://www.pjreddie.com/media/files/mnist_test.csv

