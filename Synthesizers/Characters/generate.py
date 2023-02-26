from PIL import Image, ImageDraw, ImageFont
import os
import shutil 
import random
import json
from tqdm import tqdm
from Synthesizers.utils import crop

dst = 'Datasets/SyntheticCharacters/all'
if os.path.exists(dst):
    shutil.rmtree(dst)
os.mkdir(dst)

graphemes_dict = 'Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json'
graphemes_dict = json.load(open(graphemes_dict, 'r', encoding='utf-8'))

fonts_root = 'Synthesizers/fonts'
font_files = os.listdir(fonts_root)
font_files.sort()

# some fonts have problems with some graphemes. for example font 14 has problems with 
# grapheme 4,9,10,14,17,20,21,23,25. so we skip them
problems = {1: [], 2: [], 3: [20,21,22,23,24,25], 4: [], 5: [], 6: [219], 7: [], 
            8: [219], 9: [], 10: [], 11: [], 12: [23], 13: [219], 
            14: [4,9,10,14,17,20,21,23,25], 15: [23], 16: [23], 
            17: [4,9,10,14,17,20,21,23,25], 18: [], 19: [], 20: [], 21: [], 
            22: [], 23: [219], 24: []}

for grapheme, grapheme_id in tqdm(graphemes_dict.items()):
    for _ in range(10):
        for font_file in font_files:
            font_code = int(font_file.split('.')[0])
            if grapheme_id in problems[font_code]:
                continue

            b_color = random.randint(175, 255)
            t_color = random.randint(0, 75)

            image = Image.new("L", (300, 100), (b_color))
            font_size = random.randint(20, 40)
            font = ImageFont.truetype(os.path.join(fonts_root, font_file), font_size)
            draw = ImageDraw.Draw(image)
            draw.text((30, 20), grapheme, font=font, fill=(t_color))
            image = crop(image, b_color)

            id = str(random.randint(1, 1e6)).zfill(6)
            image.save(f"{dst}/{str(grapheme_id).zfill(3)}_{font_file.split('.')[0]}_{id}.png")