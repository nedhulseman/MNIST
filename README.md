# MNIST

The purpose of this project is to model human hand-writing using basic modeling techniques. I want to show that you don't always need advanced modeling techniques like neural nets for computer vision.


## Getting Started

Using these instructions will give you a basic idea of how the script works. If you want to go dive deeper into my methodology make sure to check out the blog post at https://nedhulseman.wordpress.com/2017/11/12/using-machine-learning-techniques-to-read-hand-written-numbers/

### Prerequisites

Below is what you will need to run this script

```
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import pickle ```




## Deployment

Run through the project in order of what appears first. The first chunk splits the data into test and training. Then the script goes through and models a knn model without tuning. Next we look at PCA and model the knn using PCA. Finally we can pickle the model for safe keeping!


## Authors

@Ned Hulseman


