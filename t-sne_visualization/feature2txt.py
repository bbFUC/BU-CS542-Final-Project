import os
import numpy as np
import pandas as pd

path="./fixed"
filenames=os.listdir(path)

#filename="pig.csv"

c=1
for filename in filenames:
    print("c="+str(c))
    c+=1
    df=pd.read_csv(path+"/"+filename)
    list1=df['feature'].tolist()
    f1=open("temp.txt","w")
    print(list1,file=f1)
    f1.close()

    f2=open("temp.txt","r")
    str1=f2.read()
    f2.close()
    str2=str1.replace("'","").replace(";",",").replace("\n","")

    f3=open("./feature_matrix/"+filename.replace("csv","txt"),"w")
    f3.write(str2)
    f3.close()

    list2=df['filename'].tolist()

    f11=open("temp.txt","w")
    print(list2,file=f11)
    f11.close()

    f22=open("temp.txt","r")
    str11=f22.read()
    f22.close()
    strFile=filename.replace(".csv","")
    strPath="/mnt/udata/project/quickDraw/new_fulldata/"+strFile+"/"+strFile+"/"
    str22=str11.replace(strPath,"").replace("\n","")

    # f33=open("./filename_list/"+filename.replace("csv","txt"),"w")
    # f33.write(str22)
    # f33.close()
