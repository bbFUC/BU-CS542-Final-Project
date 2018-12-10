#coding=utf-8
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random


def drawByNdjson(raw_dataPath, categoryName, randomSample=0, start=0, end=10, scale_length=2.56, scale_width=2.56, savingPath="./"):
    f = open(raw_dataPath)
    setting = json.load(f)
    print("the number of elements in " + categoryName + " is: ")
    print(len(setting))
    #iterate all elements in the category
    if end == 0:
        end = len(setting) + 1

    if randomSample == 0:
        count = start
        for j in range(start, end):
            #check if the current draw is recognized
            if (setting[j]['recognized']==False):
                continue

            #更改画图的size plot默认(8,6)即800*600的dimension    
            plt.figure(figsize=(scale_length, scale_width))
            plt.ioff()

            #利用plot画一条又一条线
            for i in range(0,len(setting[j]['drawing'])):
                x = setting[j]['drawing'][i][0]
                y = setting[j]['drawing'][i][1]
                plt.plot(x,y,'k')
            #让图紧贴左右上下边缘
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            #直接画出来的原图是完全反转的需要更改坐标轴
            ax = plt.gca()
            #将x轴置于最上方
            ax.xaxis.set_ticks_position('top')
            #反转y坐标轴  
            ax.invert_yaxis()
            #设置横纵坐标单位长相等 否则图会失真
            ax.set_aspect(1)
            #是否显示坐标轴
            plt.axis('off')

            #将图片存起来
            plt.savefig(savingPath + "/%d.png"%count)
            plt.close()
            print(str(count) + ".png" + " in " + categoryName + " has been stored!")
            count += 1
    
    else:
        resultList=random.sample(range(0,len(setting)),end-start)
        count = 0
        for j in resultList:
            #check if the current draw is recognized
            if (setting[j]['recognized']==False):
                continue

            plt.ioff()
            #更改画图的size plot默认(8,6)即800*600的dimension    
            plt.figure(figsize=(scale_length, scale_width))

            #利用plot画一条又一条线
            for i in range(0,len(setting[j]['drawing'])):
                x = setting[j]['drawing'][i][0]
                y = setting[j]['drawing'][i][1]
                plt.plot(x,y,'k')
            #让图紧贴左右上下边缘
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            #直接画出来的原图是完全反转的需要更改坐标轴
            ax = plt.gca()
            #将x轴置于最上方
            ax.xaxis.set_ticks_position('top')
            #反转y坐标轴  
            ax.invert_yaxis()
            #设置横纵坐标单位长相等 否则图会失真
            ax.set_aspect(1)
            #是否显示坐标轴
            plt.axis('off')

            #将图片存起来
            plt.savefig(savingPath + "/%d.png"%count)
            plt.close()
            print(str(count) + ".png" + " in " + categoryName + " has been stored!")
            count += 1

    f.close()


def mkdir(path):
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder: " + path + "---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")


if __name__ == '__main__':
    ##########################
    ######input arguments#####
    ##########################
    start = 0    # the index of the first draw to be generated
    end = 15000     # the index of the last draw to be generated, if end=0 then iterate all

    #the dimension of the draw would be 256*256 in default case
    scale_length = 2.56      # the length of the draws 
    scale_width = 2.56       # the width of the draws

    #whether generate the draws randomly, if random = 1,
    #the number of generated draws should be (end - start) 
    randomSample = 1

    #files paths
    savingPath = "/usr4/cs542/liweixi/quickdraw_visualize/"    # path of saving quickdraws
    categoryPath = '/usr4/cs542/liweixi/categories.txt'  # path of the quickdraws category
    datasetPath = '/usr4/cs542/liweixi/dataset_path/'    # path of the quickdraw raw datasets

    #category filters, default is [], which means visualize all the categories
    target_category = []    # only visualize the quickdraws of these categories
    ##########################

    mkdir(savingPath)

    #default case, target_category is empty
    if target_category==[]:
        with open(categoryPath, 'rt') as f:
            for category in f:
                category = category.strip('\n')
                raw_dataPath = datasetPath + category + ".json"
                category_savingPath = savingPath + category
                mkdir(category_savingPath)
                drawByNdjson(raw_dataPath, category, randomSample, start, end, scale_length, scale_width, category_savingPath)
                print("finish visualizing category: " + category)

    else:
        for category in target_category:
                raw_dataPath = datasetPath + category + ".json"
                category_savingPath = savingPath + category
                mkdir(category_savingPath)
                drawByNdjson(raw_dataPath, category, randomSample, start, end, scale_length, scale_width, category_savingPath)
                print("finish visualizing category: " + category)