import os
import numpy as np
import pandas as pd

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x),axis=1)
    return ex/sum_ex[:,None]

feature_path="./feature_matrix"
filename_path="./filename_list"


file_class_dic=open("./class_dic.txt","r")
str_class_dic=file_class_dic.read()
file_class_dic.close()
class_dic=eval(str_class_dic)

c=0
#filename="pig.txt"
filenames=os.listdir(filename_path)
for filename in filenames:
    c=c+1
    classname=filename.replace(".txt","")
    print("processing class:"+classname+",c="+str(c))

    file_features=open(feature_path+"/"+filename,"r")
    str_features=file_features.read()
    file_features.close()
    featureList=eval(str_features)

    featureMatrix=np.array(featureList)
    numOfFeatures=featureMatrix.shape[0]
    #maxindexes=np.argmax(featureMatrix,axis=1)
    smFeatures=softmax(featureMatrix)
    groundTruth=np.zeros(345)
    groundTruth[class_dic[classname]]=1.0
    distance2gt=np.sqrt(np.sum(np.power(smFeatures-groundTruth,2),axis=1))

    sortedDist2gt=np.sort(distance2gt)
    core_th=sortedDist2gt[round(numOfFeatures*0.05)]

    ind=np.where(distance2gt<=core_th)
    core_mean=np.mean(smFeatures[ind[0],:],axis=0)

    distance2core=np.sqrt(np.sum(np.power(smFeatures-core_mean,2),axis=1))
    sortedDist2core=np.sort(distance2core)

    newInd=np.where(distance2core<=sortedDist2core[99])
    top100_smFeatures=smFeatures[newInd[0],:]
    top100_featureMatrix=featureMatrix[newInd[0],:]

    top100_sm_file=open("./top100_feature_softmax/"+classname+".txt","w")
    top100_ori_file=open("./top100_feature_ori/"+classname+".txt","w")
    print(top100_smFeatures.tolist(),file=top100_sm_file)
    print(top100_featureMatrix.tolist(),file=top100_ori_file)
    top100_sm_file.close()
    top100_ori_file.close()
