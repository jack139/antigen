import os
import glob
import json
import cv2
from tqdm import tqdm

origi_pic_path = '/home/tao/Downloads/antigen/data'
target_pic_path = 'data/crop'
json_path = '/home/tao/Downloads/antigen/json'

file_list = glob.glob(origi_pic_path+'/*')

for f in tqdm(file_list):
    # 从json文件获取CT的坐标
    basename = os.path.basename(f)

    json_file = os.path.join(json_path, os.path.splitext(basename)[0]+'.json')
    with open(json_file) as fp:
        j = json.load(fp)

    if j['shapes'][0]['label']=='CT' and j['shapes'][1]['label']=='S':
        p1 = j['shapes'][0]['points']
    elif j['shapes'][0]['label']=='S' and j['shapes'][1]['label']=='CT':
        p1 = j['shapes'][1]['points']
    else:
        print('label ERROR: ', f)
        continue

    # 截图
    image = cv2.imread(f)
    a = int(p1[0][0]) # x start
    b = int(p1[1][0]) # x end
    c = int(p1[0][1]) # y start
    d = int(p1[1][1]) # y end
    cropImg = image[c:d, a:b]
    cv2.imwrite(os.path.join(target_pic_path, basename), cropImg)


print(len(file_list))