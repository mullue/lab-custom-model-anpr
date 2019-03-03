#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import json

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized
NUMS =['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ']
CHARS=['가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '고', '노', '도', '로', '모', '구', '누', '두', '루', '무', 
       '버', '서', '어', '저', '처', '커', '터', '퍼', '보', '소', '오', '조', '초', '코', '토', '포', '부', '수', '우', '주', '추', '쿠', '투', '푸', '후', 
       '그', '느', '드', '르', '므', '브', '스', '으', '즈', '츠', '크', '트', '프', '흐',
       '바', '사', '아', '자', '차', '카', '타', '파', '하', '허', '호']
JOIN =NUMS + CHARS
n_chr=len(JOIN)

OUTPUT_SHAPE = (64, 128)
CLASSES=['License Plate']

def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in JOIN)

    for c in JOIN:
        #print(c + ';' + str(height))
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    return "{}{}{} {}{}{}{}".format(
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        CHARS[random.randrange(0,len(CHARS))],
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS))


def rounded_rect(shape, radius):
    out = np.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = round(random.uniform(0.3, 0.4) * font_height)
    v_padding = round(random.uniform(0.1, 0.3) * font_height)
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in JOIN)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width/8 + h_padding * 2))

    text_color, plate_color = pick_colors()
    text_mask = np.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (np.ones(out_shape) * plate_color * (1. - text_mask) +
             np.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=0.5,
                            scale_variation=1.5,
                            translation_variation=0.7)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    
    # generate outfile
    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += np.random.normal(scale=0.05, size=out.shape)
    out = np.clip(out, 0., 1.)
    
    # adding bounding box
    rowsum = np.sum(plate_mask,axis=1)
    colsum = np.sum(plate_mask,axis=0)
    x1=int(min(np.where(colsum)[0]))
    x2=int(max(np.where(colsum)[0]))
    y1=int(min(np.where(rowsum)[0]))
    y2=int(max(np.where(rowsum)[0]))

    #bbx = [['0', 1.0, x1, y1, x2, y2]]
    bbx = {'class_id':'0', 'prob':1.0, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}

    return out, code, bbx, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)
    
def write_files(file_id, im, c, bbx): 
    img_file = "gen-imgs/{}.png".format(file_id)
    tag_file = "gen-tags/{}.json".format(file_id)
    crp_file = "cropped-imgs/{}.png".format(file_id)
    num_file = "nums-tags/{}.json".format(file_id)
    
    # write original image file
    cv2.imwrite(img_file, im * 255.)
    # write tags json file
    label = {'file': file_id + str('.png'),
             'image_size':[
                {'width': im.shape[1],
                 'height': im.shape[0],
                 'depth': 3}],
             'annotations':[
                {'class_id': bbx['class_id'],
                 'left': bbx['x1'],
                 'top': bbx['y1'],
                 'width': bbx['x2']-bbx['x1'],
                 'height': bbx['y2']-bbx['y1'] }],
             'categories':[
                {'class_id': 0,
                 'name': CLASSES[0]}]} 
    with open(tag_file, 'w') as outfile: json.dump(label, outfile)
    # write cropped image file
    img = Image.fromarray(im*255).convert("L")
    cropped = img.crop((bbx['x1'], bbx['y1'], bbx['x2'], bbx['y2'])) 
    cropped.save(crp_file)
#     cv2.imwrite(crp_file, cropped)
    #write number plate json file
    nums = []
    for id in range(len(c)):
        char = c[id:id+1]
        nums.append(JOIN.index(char))
    nums_tag={"file": file_id + str('.png'), "nums": nums}
    with open(num_file, 'w') as outfile: json.dump(nums_tag, outfile)   
        
if __name__ == "__main__":
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, bbx, p) in enumerate(im_gen):
        file_id = "{:08d}_{}_{}_{}".format(img_idx, c[:2],c[3:], "1" if p else "0")
        write_files(file_id, im, c, bbx)
        if (img_idx<10): print(file_id)

    print('... {} files and tags were generated'.format(img_idx+1))
