import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn import manifold, datasets


filePath = '/Users/liweixi/Desktop/feature_matrix/'
# file2 = '/Users/liweixi/Desktop/feature_matrix/bird.txt'

all_files = os.listdir(filePath)
count = 0
f_tem = open(filePath + all_files[0], 'r')
str_tem = f_tem.read()
f_tem.close()
featureList = eval(str_tem)
featureMatrix = np.array(featureList)
target = np.zeros((featureMatrix.shape[0]))
previous_length = featureMatrix.shape[0]
target_length = 0
for everyFile in os.listdir(filePath):
    print("now process #%d file..." %count)
    if count == 0:
        count += 1
        continue
    f_tem = open(filePath + everyFile, 'r')
    str_tem = f_tem.read()
    f_tem.close()
    featureList = eval(str_tem)
    featureMatrix1 = np.array(featureList)
    featureMatrix = np.vstack((featureMatrix, featureMatrix1))
    target_length = featureMatrix1.shape[0]
    target_tem = np.zeros((target_length))
    target = np.append(target, target_tem)
    target[previous_length:featureMatrix.shape[0]] = count
    # update temp variable
    previous_length = featureMatrix.shape[0]
    count += 1

n_components = 2
X = featureMatrix
color = target

tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, alpha=0.3)
plt.savefig('./images/340classes.png', dpi=120)
plt.show()