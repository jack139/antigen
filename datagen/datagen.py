from os import path, mkdir
from PIL import Image
import numpy as np
from tqdm import tqdm
import math
import json

output_folder = "data/generated"
output_json_folder = "data/json"

resize_ratio = 2 # 原始图片缩小倍数

if not path.exists(output_folder):
    mkdir(output_folder)
if not path.exists(output_json_folder):
    mkdir(output_json_folder)

# 出现概率
backgrounds = [
    "1", "2", "2a", "3", "3a", 
    "4", "4a", "5", "6", "7", 
    "8", "5a", "6a", "7a", "8a"
]
backgrounds_p = [
    0.09, 0.09, 0.09, 0.09, 0.09,
    0.02, 0.02, 0.02, 0.02, 0.02,
    0.09, 0.09, 0.09, 0.09, 0.09,
]
characters = [
    "neg1", "fal1", "nul1", "pos1",
    "neg2", "fal2", "nul2", "pos2",
    "neg3", "fal3", "nul3", "pos3",
    "neg4", "fal4", "nul4", "pos4",
]
characters_p = [0.0625]*16
objects = ["none", "hand1-R", "hand2-L", "hand3-D", "hand4-L",  "hand5-R",  "hand6-U",  "hand7-L"]
objects_p = [0.4, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1]
angels = [0, 90, 180, 270]
angels_p = [0.25, 0.25, 0.25, 0.25]


# 旋转坐标（顺时针）
# 平面上一点x1,y1,绕平面上另一点x2,y2顺时针旋转θ角度
# x=(x1-x2)cosθ-(y1-y2)sinθ+x2
# y=(y1-y2)cosθ+(x1-x2)sinθ+y2
def rotate_xy(x1, y1, theta, character_size):
    if theta==90 or theta==270:
        x2 = y2 = 0
    else:
        x2 = character_size[0]/2
        y2 = character_size[1]/2

    x = (x1-x2) * math.cos(math.radians(theta)) + (y1-y2) * math.sin(math.radians(theta)) + x2
    y = (y1-y2) * math.cos(math.radians(theta)) - (x1-x2) * math.sin(math.radians(theta)) + y2

    if theta==90:
        y += character_size[1]
    elif theta==270:
        x += character_size[0]

    return x, y

# 生成 json
def generate_json(character, background_size, character_size, coordinates, rotate_angle, generated_filename):
    # 准备标记
    json_file = path.join('box_json', f'{character}.json')
    if not path.exists(json_file):
        print('read json ERROR!')
        return None

    with open(json_file) as fp:
        j = json.load(fp)

    j['imageWidth'] = background_size[0]
    j['imageHeight'] = background_size[1]

    if j['shapes'][0]['label']=='box' and j['shapes'][1]['label']=='CT':
        p1 = j['shapes'][0]['points']
        p2 = j['shapes'][1]['points']
    elif j['shapes'][0]['label']=='CT' and j['shapes'][1]['label']=='box':
        p1 = j['shapes'][1]['points']
        p2 = j['shapes'][0]['points']
    else:
        print('label err! ', json_file)
        return None

    p1[0][0] //= resize_ratio
    p1[0][1] //= resize_ratio
    p1[1][0] //= resize_ratio
    p1[1][1] //= resize_ratio

    # 保证左上、右下两个点
    if rotate_angle==0 or rotate_angle==180:
        p1[0][0], p1[0][1] = rotate_xy(p1[0][0], p1[0][1], rotate_angle, character_size)
        p1[1][0], p1[1][1] = rotate_xy(p1[1][0], p1[1][1], rotate_angle, character_size)
    else:
        tmp_x, tmp_y = rotate_xy(p1[0][0], p1[0][1], rotate_angle, character_size)
        p1[0][0], p1[1][1] = rotate_xy(p1[1][0], p1[1][1], rotate_angle, character_size)
        p1[1][0], p1[0][1] = tmp_x, tmp_y

    p1[0][0] += coordinates[0]
    p1[0][1] += coordinates[1]
    p1[1][0] += coordinates[0]
    p1[1][1] += coordinates[1]

    p2[0][0] //= resize_ratio
    p2[0][1] //= resize_ratio
    p2[1][0] //= resize_ratio
    p2[1][1] //= resize_ratio

    # 保证左上、右下两个点
    if rotate_angle==0 or rotate_angle==180:
        p2[0][0], p2[0][1] = rotate_xy(p2[0][0], p2[0][1], rotate_angle, character_size)
        p2[1][0], p2[1][1] = rotate_xy(p2[1][0], p2[1][1], rotate_angle, character_size)
    else:
        tmp_x, tmp_y = rotate_xy(p2[0][0], p2[0][1], rotate_angle, character_size)
        p2[0][0], p2[1][1] = rotate_xy(p2[1][0], p2[1][1], rotate_angle, character_size)
        p2[1][0], p2[0][1] = tmp_x, tmp_y

    p2[0][0] += coordinates[0]
    p2[0][1] += coordinates[1]
    p2[1][0] += coordinates[0]
    p2[1][1] += coordinates[1]

    # 调整坐标顺序
    if p1[0][0]>p1[1][0]:
        p1[0], p1[1] = p1[1], p1[0]

    if p2[0][0]>p2[1][0]:
        p2[0], p2[1] = p2[1], p2[0]

    j['imagePath'] = f'../generated/{generated_filename}'

    base_filename = path.splitext(generated_filename)[0]

    json.dump(
        j,
        open(f'{output_json_folder}/{base_filename}.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )    

    return j

def generate_image(background, character, object, file_name):
    """Generate image with given background, given character and given object and save it with the given file name

    Args:
        background (str): background name
        character (str): character name
        object (str): object name
        file_name (str): file name
    """
    background_file = path.join("backgrounds", f"{background}.png")
    background_image = Image.open(background_file)

    (width, height) = (background_image.width // resize_ratio, background_image.height // resize_ratio)
    background_image = background_image.resize((width, height))

    #Create character
    character_file = path.join("characters", f"{character}.png")
    character_image = Image.open(character_file)

    (width, height) = (character_image.width // resize_ratio, character_image.height // resize_ratio)
    character_image = character_image.resize((width, height))

    # rotate
    rotate_angle = np.random.choice(np.arange(0,len(angels)), p=angels_p)
    rotate_angle = angels[rotate_angle]
    character_image = character_image.rotate(rotate_angle, expand=True)

    xx = background_image.width//2-character_image.width//2
    yy = background_image.height//2-character_image.height//2

    coordinates = (
        xx + np.random.randint(-xx//2, xx//2),
        yy + np.random.randint(-yy//2, yy//2)
        #np.random.randint(0,background_image.width-character_image.width), 
        #np.random.randint(0,background_image.height-character_image.height)
    ) #x, y

    background_image.paste(character_image, coordinates, mask=character_image)


    #Create object
    if object != "none":
        object, object_pos = object.split('-')
        object_file = path.join("objects", f"{object}.png")
        object_image = Image.open(object_file)

        (width, height) = (object_image.width // resize_ratio, object_image.height // resize_ratio)
        object_image = object_image.resize((width, height))


        if object_pos=='R': # 在右侧
            if rotate_angle==0 or rotate_angle==180:
                x_offset = character_image.width+np.random.randint(-5, 0)
                y_offset = np.random.randint(-80, -50)
            else:
                x_offset = character_image.width+np.random.randint(-50, -30)
                y_offset = np.random.randint(-80, -50)
        elif object_pos=='L': # 在左侧
            if rotate_angle==0 or rotate_angle==180:
                x_offset = -object_image.width+np.random.randint(0, 5)
                y_offset = np.random.randint(-80, -50)
            else:
                x_offset = -object_image.width+np.random.randint(30, 50)
                y_offset = np.random.randint(-80, -50)
        elif object_pos=='U': # 在上面
            x_offset = np.random.randint(-80, -30)
            y_offset = -object_image.height+np.random.randint(0, 5)
        else: # 在下面
            x_offset = np.random.randint(-80, -30)
            y_offset = character_image.height+np.random.randint(-5, 0)

        coordinates2 = (coordinates[0]+x_offset, coordinates[1]+y_offset) #x, y

        background_image.paste(object_image, coordinates2, mask=object_image)

    file_name = f"{file_name}.jpg"
    output_file = path.join(output_folder, file_name)
    background_image.save(output_file)

    generate_json(character, background_image.size, character_image.size, 
        coordinates, rotate_angle, file_name)


def generate_random_imgs(total_imgs):
    """Generates a given number of random images according to predefined probabilities

    Args:
        total_imgs (int): total number of images to generate
    """
    for num in tqdm(range(total_imgs)):
        background = np.random.choice(np.arange(0,len(backgrounds)), p=backgrounds_p)
        background = backgrounds[background]
        
        character = np.random.choice(np.arange(0,len(characters)), p=characters_p)
        character = characters[character]

        object = np.random.choice(np.arange(0,len(objects)), p=objects_p)
        object = objects[object]

        generate_image(background, character, object, f"{num:06d}_{character[:3]}")


if __name__ == "__main__":
    generate_random_imgs(20)

