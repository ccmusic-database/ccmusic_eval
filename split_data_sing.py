import os
from shutil import copy
import random
import sys

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
        
os.chdir(sys.path[0])
# file = "sing_data/男x女"
file = "sing_data/美声x民族"
class_tags = [cla for cla in os.listdir(file) if ".txt" not in cla]
mkfile(file + '/train')
for cla in class_tags:
    mkfile(file + '/train/' + cla)

mkfile(file + '/val')
for cla in class_tags:
    mkfile(file + '/val/' + cla)

split_rate = 0.1
for cla in class_tags:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = file + '/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = file + '/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("processing done!")