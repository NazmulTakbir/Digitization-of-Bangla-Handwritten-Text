from PIL import Image, ImageDraw, ImageFont
import os
import shutil 
from Synthesizers.utils import crop
import random

dst = 'Datasets/SyntheticWords/all'
if os.path.exists(dst):
    shutil.rmtree(dst)
os.mkdir(dst)

fonts_root = 'Synthesizers/fonts'
font_files = os.listdir(fonts_root)
font_files.sort()

lines = open('Datasets/SyntheticWords/labels.csv', 'r', encoding='utf-8').readlines()
lines = [line.strip() for line in lines]
words = [line[len(line.split(',')[0])+1:] for line in lines]

def word_generator(height, width):
    word = random.choice(words)
    font_file = random.choice(font_files)
    
    b_color = random.randint(175, 255)
    t_color = random.randint(0, 75)

    image = Image.new("L", (300, 100), (b_color))
    font_size = random.randint(20, 40)
    font = ImageFont.truetype(os.path.join(fonts_root, font_file), font_size)
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), word, font=font, fill=(t_color))
    image = crop(image, b_color, height=height, width=width, min_margin=10, max_margin=30)

    return image, word