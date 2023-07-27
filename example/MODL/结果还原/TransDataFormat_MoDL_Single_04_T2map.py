import os
import re
import scipy.io as sio
from collections import defaultdict
import numpy as np
'''
将王光明生成的数据转成Submission的标准格式
'''
path = '/media/nas/Seafile_Storage/Raw_data/MICCAIChallenge2023/ChallengeResult/Submission_MoDL/Single_Mapping_04_T2/'

pattern = r'Single_Mapping_(\d+)_([A-Za-z0-9]+)'
match = re.search(pattern, path)

if match:
    AccFactor = match.group(1)
    Sub_Type = match.group(2)

if Sub_Type == 'T1':
    pattern = r'single_rec_P(\d+)_T2map_(\d+)_(\d+)\.mat'  # 文件名的正则表达式模式
elif Sub_Type == 'T2':
    pattern = r'single_rec_P(\d+)_T2map_(\d+)_(\d+)\.mat'
else:
    print('SubType Error!')

# 创建字典来存储每个人的数据
people_info = defaultdict(list)

# 遍历目录下的所有文件
for file_name in os.listdir(path):
    match = re.match(pattern, file_name)
    if match:
        person = int(match.group(1))
        layer = int(match.group(2))
        time_point = int(match.group(3))

        # 读取.mat文件的数据
        file_path = os.path.join(path, file_name)
        data = sio.loadmat(file_path)['data']

        # 存储数据到对应人员的列表中
        people_info[person].append((layer, time_point, data))

# 存储每个人的数据到独立的文件夹中
savepath = '/media/nas/Seafile_Storage/Raw_data/MICCAIChallenge2023/ChallengeResult/Submission_MoDL/Mapping/SingleCoil/ValidationSet/AccFactor'+ AccFactor

for person, data_list in people_info.items():
    # 创建每个人的文件夹
    person_folder = os.path.join(savepath, f'P{person:03d}')
    os.makedirs(person_folder, exist_ok=True)

    # 构建每个人的数据矩阵
    num_layers = max([layer for layer, _, _ in data_list])
    num_time_points = max([time_point for _, time_point, _ in data_list])
    w, h = data_list[0][2].shape
    data_matrix = np.zeros((w, h, num_layers, num_time_points))

    for layer, time_point, data in data_list:
        data_matrix[:, :, layer-1, time_point-1] = np.abs(data)

    # 构建每个人的.mat文件完整路径
    if Sub_Type == 'T1':
        file_path = os.path.join(person_folder, 'T1map.mat')
    elif Sub_Type == 'T2':
        file_path = os.path.join(person_folder, 'T2map.mat')
    else:
        break
    # 将数据保存为.mat文件
    sio.savemat(file_path, {'data': data_matrix})

print('done!')