#coding=utf-8
import json
import csv
import os


datasetsPath = '/Users/liweixi/Downloads/123/'    # csv文件夹路径
datasetsFiles = os.listdir(datasetsPath)
jsonWritePath = '/Users/liweixi/Downloads/code/'    # json储存的路径

for file in datasetsFiles:
    num = 0
    with open(datasetsPath+file, 'r') as f1:
        for line in f1:
            num+=1
    csvFile = csv.reader(open(datasetsPath+file))
    for row in csvFile:
        field = list(row)
        break
    file = file.replace(' ', '_')
    file = file.replace('csv', 'json')
    f = open(jsonWritePath+file, 'w')
    f.write('[')
    count = 1
    length = num 
    for row in csvFile:
        if count == length-1:
            tem = '{"'+ field[1]+'":'+row[1] + ',' + '"'+field[2]+'":'+'"'+row[2] + '",' + '"'+field[3] + '":' + row[3] + '}]'
            f.write(tem)
            break
        tem =  '{"'+ field[1]+'":'+row[1] + ',' + '"'+field[2]+'":'+'"'+row[2] + '",' + '"'+field[3] + '":' + row[3] +'},'
        f.write(tem)
        count += 1
        if count % 1000 == 0:
            print(count)
    f.close()