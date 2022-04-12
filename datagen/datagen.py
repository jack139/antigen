from os import path, mkdir
from PIL import Image
import numpy as np
from tqdm import tqdm

output_folder = "data/generated"
if not path.exists(output_folder):
    mkdir(output_folder)

backgrounds = ["1", "1", "1", "1"]
backgrounds_p = [0.3, 0.3, 0.3, 0.1]
characters = ["fal1", "fal1", "fal1", "fal1", "fal1"]
characters_p = [0.4, 0.3, 0.2, 0.095, 0.005]
objects = ["none", "hand1"]
objects_p = [0.5, 0.5]
angels = [0, 90, 180, 270]
angels_p = [0.25, 0.25, 0.25, 0.25]

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

    #Create character
    character_file = path.join("characters", f"{character}.png")
    character_image = Image.open(character_file)

    # rotate
    rotate_angle = np.random.choice(np.arange(0,len(angels)), p=angels_p)
    rotate_angle = angels[rotate_angle]
    character_image = character_image.rotate(rotate_angle, expand=True)

    coordinates = (
        #int(background_image.width/2-character_image.width/2), 
        np.random.randint(0,int(background_image.width-character_image.width)), 
        np.random.randint(0,int(background_image.height-character_image.height))
    ) #x, y
    background_image.paste(character_image, coordinates, mask=character_image)


    #Create object
    if object != "none":
        object_file = path.join("objects", f"{object}.png")
        object_image = Image.open(object_file)

        if rotate_angle==0 or rotate_angle==180:
            coordinates2 = (
                int(coordinates[0]+character_image.width+np.random.randint(-20, 0)), 
                coordinates[1]+np.random.randint(-250, -150)
            ) #x, y
        else:
            coordinates2 = (
                int(coordinates[0]+character_image.width+np.random.randint(-150, -80)), 
                coordinates[1]+np.random.randint(-250, -150)
            ) #x, y
        background_image.paste(object_image, coordinates2, mask=object_image)

    output_file = path.join(output_folder, f"{file_name}.jpg")
    background_image.save(output_file)


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

        generate_image(background, character, object, f"generated{num}")


if __name__ == "__main__":
    #generate_all_imgs()
    generate_random_imgs(10)

