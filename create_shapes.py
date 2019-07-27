# create_shapes.py
# Arnav Ghosh
# 27th July 2019

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# CONSTANTS
DIM = 256
INIT_SPACE = 10
LENGTH_BOUNDS= (15, 128)
WIDTH_BOUNDS = (30, 56)
ROTATION_BOUNDS = (0, 180)

def repeat_rectangles(num_length, num_width, num_rot):
    lengths = [LENGTH_BOUNDS[0]] + list(np.random.choice(range(*LENGTH_BOUNDS), num_length, replace = False))
    widths = [WIDTH_BOUNDS[0]] + list(np.random.choice(range(*WIDTH_BOUNDS), num_width, replace = False))
    rotations = [ROTATION_BOUNDS[0]] + list(np.random.choice(range(*ROTATION_BOUNDS), num_rot, replace = False))

    for rot in rotations:
        # create rotation matrix
        angle = np.radians(rot)
        c_angle, s_angle = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c_angle, -s_angle], [s_angle, c_angle]])
        
        for leng in lengths:
            for wid in widths:
                dist_centers = np.sqrt(((leng ** 2) + (wid ** 2))) + INIT_SPACE
                x, y = dist_centers/2, DIM/2
                rot_vertices = np.matmul(rot_matrix, np.array([[leng/2, leng/2, -leng/2, -leng/2], 
                                                               [wid/2, -wid/2, -wid/2, wid/2]]))

                img = Image.new('RGB', (DIM, DIM), color = 'white')
                draw = ImageDraw.Draw(img)
                
                # until center is out of the img
                while x < DIM:
                    vertices = rot_vertices + np.array([[x], [y]])
                    draw.polygon(list(map(tuple, vertices.transpose())), fill = 'black')

                    x += dist_centers

                img.save(f"repeat_rectangle_{leng}_{wid}_{rot}.jpg", "JPEG")

# # length, width, rotation, x, y (latter determine spacing) 
# # each argument is a list indicating what happens on the 1st, 2nd third itertaion etc. wraps around
# # (all in step)
def patterned_rectangles(length_patt, width_patt, rotation_patt, x_space_patt, y_space_patt, total_num):

    img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    leng, wid, rot = LENGTH_BOUNDS[0], WIDTH_BOUNDS[0], ROTATION_BOUNDS[0]
    x_centers_dist = np.sqrt(((leng ** 2) + (wid ** 2))) + INIT_SPACE
    y_centers_dist = 0
    x, y = x_centers_dist/2, DIM/2

    for j in range(total_num): 
        i = 0
        while x < DIM:
            angle = np.radians(rot)
            c_angle, s_angle = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[c_angle, -s_angle], [s_angle, c_angle]])

            rot_vertices = np.matmul(rot_matrix, np.array([[leng/2, leng/2, -leng/2, -leng/2], 
                                                            [wid/2, -wid/2, -wid/2, wid/2]]))
            vertices = rot_vertices + np.array([[x], [y]])
            draw.polygon(list(map(tuple, vertices.transpose())), fill = 'black')

            x +=  np.sqrt(((leng ** 2) + (wid ** 2))) + x_space_patt[i % len(x_space_patt)]
            y += y_space_patt[i % len(y_space_patt)]
            leng += length_patt[i % len(length_patt)]
            wid += width_patt[i % len(width_patt)]
            rot += rotation_patt[i % len(rotation_patt)]

            i += 1

        img.save(f"pattern_rectangle_{j}.jpg", "JPEG")

# need main function that varies 1 at a time and then 2 etc.



# source_img = Image.open(file_name).convert("RGB")

# draw = ImageDraw.Draw(source_img)
# draw.rectangle(((0, 00), (100, 100)), fill="black")
# draw.text((20, 70), "something123", font=ImageFont.truetype("font_path123"))

# source_img.save(out_file, "JPEG")