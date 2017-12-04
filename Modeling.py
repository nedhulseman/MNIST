import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import pickle 



#############################################
# @Ned Hulseman                             #
# 9/23/2017                                 #
# MNIST Digit Reading                       #
#                                           #
# Data: MNIST Kaggle data                   #
# https://www.kaggle.com/c/digit-recognizer #
#############################################


# @MNIST
MNIST=pd.read_csv("C:\\Users\\nedhu\\Desktop\\nedhulseman.com\\numbers\\written_numbers_train.csv")
print(MNIST.shape) #42000, 785

# @label position:1 values:0-9
# @pixel1-pixel784

#Splits the data with a label and independent variables
MNIST_label=MNIST.iloc[:,0]
MNIST_pixels=MNIST.iloc[:, 1:len(MNIST.columns)]

#Creates training and test data for MNIST
X_train, X_test, y_train, y_test = train_test_split(MNIST_pixels, MNIST_label, test_size = .3, random_state=42)

#################################################################################################################
################################################# KNN Modeling ##################################################
knn_clf=KNeighborsClassifier(n_neighbors=9, weights='distance')
knn_clf.fit(X_train, y_train)
knn_clf.get_params()


predictions=knn_clf.predict(X_test)
accuracy=knn_clf.score(X_test, y_test)#96.5%
print(predictions)
mat=confusion_matrix(y_test, predictions)
np.fill_diagonal(mat, 0)
plt.matshow(mat)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Comparing Predictions and Actual Labels', y=1.08)
plt.show()

row_sums=confusion_matrix.sum(axis=1, keepdims=True)
normalized_confusion_matrix=confusion_matrix/row_sums
np.fill_diagonal(normalized_confusion_matrix, 0)
plt.matshow(normalized_confusion_matrix)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Comparing Actual and Predicted Proportions', y=1.08)
plt.show()


######################
# Results            # 
# 96.5, k=5          #
# 96.65 k=3          #
# 96.9, k=1          #
######################


#################################################################################################################
#################################################################################################################



#################################################################################################################
############################################## KNN Modeling with PCA ############################################



pca=PCA(n_components=400)
princomp=pca.fit(X_test)
var_explained=np.cumsum(pca.explained_variance_ratio_)


plt.plot(var_explained)
plt.ylabel('Variance Explained')
plt.xlabel('Principal Components')
plt.axhline(y=.95, color='r', ls='dotted', label='95%')
plt.show()
#About 150 PCs explain 95% of the variance
pca=PCA(n_components=150)
train_princomp=pca.fit_transform(X_train)
test_princomp=pca.transform(X_test)

knn_clf=KNeighborsClassifier(n_neighbors=1, weights='uniform')
knn_clf.fit(train_princomp, y_train)
predictions=knn_clf.predict(test_princomp)
accuracy=knn_clf.score(test_princomp, y_test)
print(predictions)

confusion_matrix=confusion_matrix(y_test, predictions)
np.fill_diagonal(normalized_confusion_matrix, 0)
plt.matshow(normalized_confusion_matrix)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Comparing Predictions and Actual Labels', y=1.08)
plt.show()

row_sums=confusion_matrix.sum(axis=1, keepdims=True)
normalized_confusion_matrix=confusion_matrix/row_sums

##################################
# Results                        #
# 96.7 PCA, k=5, weight=uni      #
# 96.8 PCA, k=5, weight=distance #
# 96.7 PCA, k=3, weight=uni      #
# 96.9 PCA, k=3, weight=distance #
# 96.9 PCA, k=1                  #
##################################


#################################################################################################################
#################################################################################################################





#################################################################################################################
####################################### KNN Modeling with PCA, Augmented Data ###################################


def shifty(df): #Creates a df with 4 shifted versions of the original df
    righty_shifted=df.copy()
    lefty_shifted=df.copy()
    up_shifted=df.copy()
    down_shifted=df.copy()
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
                righty_shifted.iloc[i, j]=df.iloc[i, j-1]
                lefty_shifted.iloc[i, j-1]=df.iloc[i, j]
                up_shifted.iloc[i-1, j]=df.iloc[i, j]
                down_shifted.iloc[i, j]=df.iloc[i-1, j]
    shifted=righty_shifted.append(lefty_shifted)
    shifted=shifted.append(up_shifted)
    shifted=shifted.append(down_shifted)
    return df.append(shifted)



duplicated_y_train=[]
for i in y_train.iloc[0:10000]:
    duplicated_y_train.extend([i, i, i, i, i])
duplicated_y_train=pd.DataFrame(duplicated_y_train)
augment_y_train=pd.DataFrame(y_train[10000:42000].append(duplicated_y_train))
foo=np.ravel(augment_y_train)

foo=X_train.iloc[0:10000, :]
augmented_training=shifty(foo)
augment_x_train=pd.DataFrame(X_train.iloc[10000:42000, :].append(augmented_training))
augment_x_train.to_csv('augmented_x_train.csv')
  

pca=PCA(n_components=150)
train_princomp=pca.fit_transform(augment_x_train)
test_princomp=pca.transform(X_test)

knn_clf=KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_clf.fit(train_princomp, augment_y_train.values.ravel())
predictions=knn_clf.predict(test_princomp)
accuracy=knn_clf.score(test_princomp, y_test)
print(predictions)

confusion_matrix=confusion_matrix(y_test, predictions)
np.fill_diagonal(normalized_confusion_matrix, 0)
plt.matshow(normalized_confusion_matrix)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Comparing Predictions and Actual Labels', y=1.08)
plt.show()

row_sums=confusion_matrix.sum(axis=1, keepdims=True)
normalized_confusion_matrix=confusion_matrix/row_sums



#################################################################################################################
#################################################################################################################




#################################################################################################################
############################################## pickling ###########################################################

s = pickle.dumps(knn_clf)
joblib.dump(knn_clf, 'MNIST_PCA_KNN_CLF.pkl') 
with open('C:\\Users\\nedhu\\Desktop\\nedhulseman.com\\numbers', 'wb') as f:
    pickle.dump(s, f)



#################################################################################################################
#################################################################################################################







