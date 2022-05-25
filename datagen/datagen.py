from os import path, mkdir
from PIL import Image, ImageFilter
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
    "10a",  "1a",  "2a",  "3a",  "4a",  "5a",  "6a",
    "7a",  "8a",  "9a",  "a3a",  "x10",  "x2a",  "x9aa",
    "10",  "1",  "2",  "3",  "4",  "5",  "6",
    "7",  "8",  "9",  "a3",  "x1",  "x2",  "x9a",
]
backgrounds_p = [1/28]*28
characters = [
    "fal1a",  "fal_4a",  "fal_7",  "neg_2",  "neg_4",  "nul_2a",  "nul3",  "nul_7a",  "pos_3aa",  "pos_4",
    "fal1",  "fal4a",  "neg_1a",  "neg_3a",  "neg_5a",  "nul2a",  "nul_4a",  "nul_7",  "pos_3ab",  "pos4",
    "fal2a",  "fal_4",  "neg1a",  "neg3a",  "neg_5",  "nul_2",  "nul4a",  "pos1a",  "pos_3a",  "pos_5a",
    "fal_3a",  "fal4",  "neg_1",  "neg_3",  "neg_7a",  "nul2",  "nul_4",  "pos_2a",  "pos3a",  "pos_5",
    "fal3a",  "fal_5a",  "neg1",  "neg3",  "neg_7",  "nul_3a",  "nul4",  "pos2a",  "pos3",  "pos_7a",
    "fal_3",  "fal_5",  "neg_2a",  "neg_4a",  "none",  "nul3a",  "nul_6a",  "pos_2",  "pos_4a",  "pos_7b",
    "fal3",  "fal_7a",  "neg2a",  "neg4a",  "nul1a",  "nul_3",  "nul_6",  "pos2",  "pos4a",  "pos_7",
    "fal_a5",  "neg_a3",  "neg_a5",  "pos_a1",  "pos_a3a",  "pos_a4",   "pos_a7a",
    "neg_a1",  "neg_a4",  "neg_a6",  "pos_a2a",  "pos_a3",  "pos_a6a",  "pos_a7",
    "neg_a2",  "neg_a5a",  "pos_a1a",  "pos_a2",  "pos_a4a",  "pos_a6",
    "neg_c5a",  "pos_c1a",  "pos_c2a",  "pos_c3a",  "pos_c4a", "pos_c5a",
    "neg_c5",  "pos_c1",  "pos_c2",  "pos_c3",  "pos_c4", "pos_c5",
    "neg_d1",  "neg_d2",  "neg_d3",  "neg_d4",  "pos_d4", 
    "neg_d5",  "neg_d6",  "neg_d7",  "pos_d8", "neg_d9", "pos_d9",
    "neg_e3",  "neg_e4",  "pos_e1b",  "pos_e1",  "pos_e2b",  "pos_e2",

    # 2022-05-25
    "neg_6a1",  "neg_6a4",  "neg_6b3",  "neg_6c3",  "neg_6d2",  "neg_6e1",  "neg_6g2",  "nul_6d2",  "pos_6c2",
    "neg_6a2",  "neg_6b1",  "neg_6b4",  "neg_6c4",  "neg_6d3",  "neg_6f1",  "neg_6h1",  "pos_6a1",  "pos_6d3",
    "neg_6a3",  "neg_6b2",  "neg_6c2",  "neg_6d1",  "neg_6d4",  "neg_6g1",  "nul_6b2",  "pos_6b1",  "pos_6h1",
]
characters_p = [1/146]*146
objects = [
    "none", "hand1-R", "hand2-L", "hand3-D", "hand4-L",  "hand5-R",  "hand6-U",  "hand7-D",
    "id1-D", "id2-R", "id3-U", "id4-D", "id5-U", "id6-R", "id7-U", "id8-U", "id9-D", "id10-D",
    "id1-U", "id2-L", "id3-D", "id4-U", "id5-D", "id6-L", "id7-D", "id8-D", "id9-U", "id10-U",
]
objects_p = [0.2] + [0.8/27]*27
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

    if j['shapes'][0]['label']=='box':
        p1 = j['shapes'][0]['points']
    else:
        print('label err! ', json_file)
        return None

    if character=='none':
        p1[0], p1[1] = [0, 0], [0, 0]

    else:
        p1[0][0] //= resize_ratio
        p1[0][1] //= resize_ratio
        p1[1][0] //= resize_ratio
        p1[1][1] //= resize_ratio

        p1[0][0], p1[0][1] = rotate_xy(p1[0][0], p1[0][1], rotate_angle, character_size)
        p1[1][0], p1[1][1] = rotate_xy(p1[1][0], p1[1][1], rotate_angle, character_size)

        p1[0][0] += coordinates[0]
        p1[0][1] += coordinates[1]
        p1[1][0] += coordinates[0]
        p1[1][1] += coordinates[1]

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

    # 背景随机收缩 2、3、4倍，长宽分开缩放，会出现细长条
    background_x_ration = np.random.randint(2, 5)
    background_y_ration = np.random.randint(2, 5)
    (width, height) = (background_image.width // background_x_ration, background_image.height // background_y_ration)
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
        xx + np.random.randint(-abs(xx), abs(xx)+1), ## 可能为 负数
        yy + np.random.randint(-abs(yy), abs(yy)+1)
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
                x_offset = character_image.width+np.random.randint(-20, 0)
                y_offset = np.random.randint(-50, -10)
            else:
                x_offset = character_image.width+np.random.randint(-character_image.width//3, 0)
                y_offset = np.random.randint(-50, -10)
        elif object_pos=='L': # 在左侧
            if rotate_angle==0 or rotate_angle==180:
                x_offset = -object_image.width+np.random.randint(0, 20)
                y_offset = np.random.randint(-50, -10)
            else:
                x_offset = -object_image.width+np.random.randint(0, character_image.width//3)
                y_offset = np.random.randint(-50, -10)
        elif object_pos=='U': # 在上面
            if rotate_angle==0 or rotate_angle==180:
                x_offset = np.random.randint(-80, -30)
                y_offset = -object_image.height+np.random.randint(0, character_image.height//3)
            else:
                x_offset = np.random.randint(-80, -30)
                y_offset = -object_image.height+np.random.randint(0, 5)
        else: # 在下面
            if rotate_angle==0 or rotate_angle==180:
                x_offset = np.random.randint(-80, -30)
                y_offset = character_image.height+np.random.randint(-character_image.height//3, 0)
            else:
                x_offset = np.random.randint(-80, -30)
                y_offset = character_image.height+np.random.randint(-5, 0)

        coordinates2 = (coordinates[0]+x_offset, coordinates[1]+y_offset) #x, y

        background_image.paste(object_image, coordinates2, mask=object_image)

    # 噪声
    background_image = background_image.filter(ImageFilter.GaussianBlur(np.random.randint(2)))

    file_name = f"{file_name}.jpg"
    output_file = path.join(output_folder, file_name)
    background_image.save(output_file)

    generate_json(character, background_image.size, character_image.size, 
        coordinates, rotate_angle, file_name)


def generate_random_imgs(prefix, total_imgs):
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

        generate_image(background, character, object, f"{prefix}_{num:06d}_{character[:3]}")

    return 'ok'

if __name__ == "__main__":
    #generate_random_imgs('test', 20)

    # 多线程生成图片
    params = [ # prefix, image_count
        (1, 250),
        (2, 250),
        (3, 250),
        (4, 250),
        (5, 250),
        (6, 250),
        (7, 250),
        (8, 250),
    ]

    import concurrent.futures
    import urllib.request

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(params)) as executor:
        future_to_gemerate = {executor.submit(generate_random_imgs, p[0], p[1]): p for p in params}
        for future in concurrent.futures.as_completed(future_to_gemerate):
            f = future_to_gemerate[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (f, exc))
            else:
                print('%r returned: %s' % (f, data))
