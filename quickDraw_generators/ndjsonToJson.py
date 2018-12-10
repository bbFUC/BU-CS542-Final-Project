import os


datasetsPath = '/Users/liweixi/Downloads/quickDrawDataset/'
datasetsFiles = os.listdir(datasetsPath)
jsonWritePath = '/Users/liweixi/Downloads/dataset_path/'

for file in datasetsFiles:
    with open(datasetsPath+file, 'r') as f_read:
        file = file.replace(' ', '_')
        file = file.replace('ndjson', 'json')
        f_write = open(jsonWritePath+file, 'w')
        f_write.write('[')
        all_lines = f_read.readlines()
        sub_lines = all_lines[:-1]
        for line in sub_lines:
            line = line.strip('\n')
            f_write.write(line+',')
        last_line = all_lines[-1].strip('\n')
        f_write.write(last_line+']')
        f_write.close()