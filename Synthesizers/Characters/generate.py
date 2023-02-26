from PIL import Image, ImageDraw, ImageFont
import os
import shutil 
import numpy as np
import random

if os.path.exists('Images'):
    shutil.rmtree('Images')
os.mkdir('Images')

f = open('graphemes.txt', 'r', encoding='utf-8').readlines()
f = [x[:-1] for x in f]

font_files = os.listdir('fonts')
font_files.sort()

def crop(img, b_color):
    """Crops around the character. Space around the character is random."""
    img = np.array(img)

    first_row = np.where(img.sum(axis=1) < b_color * img.shape[1])[0][0]
    last_row = np.where(img.sum(axis=1) < b_color * img.shape[1])[0][-1]
    first_col = np.where(img.sum(axis=0) < b_color * img.shape[0])[0][0]
    last_col = np.where(img.sum(axis=0) < b_color * img.shape[0])[0][-1]

    first_row = max(0, first_row-random.randint(0, 20))
    first_col = max(0, first_col-random.randint(0, 20))
    last_row = min(img.shape[0], last_row+random.randint(0, 20))
    last_col = min(img.shape[1], last_col+random.randint(0, 20))

    img = img[first_row:last_row, first_col:last_col]
    img = Image.fromarray(img)
    img = img.resize((32, 32))

    return img

# some fonts have problems with some graphemes. for example font 3 has problems with 
# grapheme 16, 17, 18, 19. so we skip them
problems = {1: [], 2: [], 3: [16,17,18,19], 4: [], 5: [], 6: [165], 7: [], 
            8: [165], 9: [], 10: [], 11: [], 12: [18], 13: [165], 
            14: [3,7,8,12,14,16,17,18,19], 15: [18], 16: [18], 
            17: [3,7,8,12,14,16,17,18,19], 18: [], 19: [], 20: [], 21: [], 
            22: [], 23: [165], 24: []}

# the following graphemes are written to a picture properly in windows OS. In linux
# they are written by unwanted "dotted circular regions"
# the images for these graphemes were generated separately in windows
windows_only = [158, 164, 156, 160, 157, 22, 159, 153, 155, 161, 154, 152, 163, 162, 23, 21]

for _ in range(10):
    for i, char in enumerate(f, 1):
        for font_file in font_files:
            if i in windows_only:
                continue
            font_code = int(font_file.split('.')[0])
            if i in problems[font_code]:
                continue

            b_color = random.randint(175, 255)
            t_color = random.randint(0, 75)

            image = Image.new("L", (300, 100), (b_color))
            font_size = random.randint(20, 40)
            font = ImageFont.truetype(os.path.join('fonts', font_file), font_size)
            draw = ImageDraw.Draw(image)
            draw.text((30, 20), char, font=font, fill=(t_color))
            image = crop(image, b_color)

            id = str(random.randint(1, 1e6)).zfill(6)
            image.save(f"Images/{str(i).zfill(3)}_{font_file.split('.')[0]}_{id}.png")