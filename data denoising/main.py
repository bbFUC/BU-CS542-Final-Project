import os
import numpy as np
import pandas as pd

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x),axis=1)
    return ex/sum_ex[:,None]

full_data_path="/mnt/udata/project/quickDraw/new_fulldata"
feature_path="./feature_matrix"
filename_path="./filename_list"
version_path="./versions"

file_class_dic=open("./class_dic.txt","r")
str_class_dic=file_class_dic.read()
file_class_dic.close()
class_dic=eval(str_class_dic)

c=0
#filename="pig.txt"
filenames=os.listdir(filename_path)
for filename in filenames:
    c=c+1
    if c<=321: #
        continue
    classname=filename.replace(".txt","")
    print("processing class:"+classname)

    file_filenames=open(filename_path+"/"+filename,"r")
    str_filenames=file_filenames.read()
    file_filenames.close()
    filenameList=np.array(eval(str_filenames))

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

    percents=[0.01,0.02,0.05,0.1,0.2,0.5]
    version_ths=[sortedDist2core[round(numOfFeatures*per)] for per in percents ]

    #version_ID=0
    for version_ID in range(len(percents)):
        print("class_ID="+str(c)+"; classname="+classname+"; version_ID="+str(version_ID))
        version_th=version_ths[version_ID]

        version_classFolder=version_path+"/version_"+str(round(100*percents[version_ID]))+"/"+classname
        os.makedirs(version_classFolder)
        newInd=np.where(distance2core<=version_th)

        selectedFiles=filenameList[newInd[0]].tolist()

        for fName in selectedFiles:
            os.symlink(full_data_path+"/"+classname+"/"+classname+"/"+fName,version_classFolder+"/"+fName)

