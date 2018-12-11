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

    per1=1-0.01
    per2=1-0.1
    #drop_1=np.random.choice(df.index, round(len(df.index)*per1),replace=False)
    #drop_2=np.random.choice(df.index, round(len(df.index)*per2),replace=False)
    drop_3=np.random.choice(df.index, len(df.index)-100,replace=False)
    #new_df_1=df.drop(drop_1)
    #new_df_2=df.drop(drop_2)
    new_df_3=df.drop(drop_3)

    #new_df_1.to_csv("./new/"+filename)
    #new_df_2.to_csv("./new2/"+filename)
    new_df_3.to_csv("./new4/"+filename)