import os

path="./class_feature_tensor"
filenames=os.listdir(path)

c=1
for filename in filenames:
    print("c="+str(c))
    c+=1
    ori_file=open(path+"/"+filename,"r")

    ori_str=ori_file.read()
    new_str=ori_str.replace('\n','').replace(']',']\n').replace('feature','feature\n')

    new_file=open("./new/"+filename,"w")
    new_file.write(new_str)
    new_file.close()
