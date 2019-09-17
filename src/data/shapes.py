# create_shapes.py
# Arnav Ghosh
# 12 Aug. 2019

from shape_constants import *

import copy
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter

########### MAIN ###########

def create_dataset(root):

    # repeating rectangle patterns
    print('Creating Rectangle Patterns')
    rect_pth = os.path.join(root, "rectangle_patterns")
    if not os.path.isdir(rect_pth):
        os.mkdir(rect_pth)

        data_pth = os.path.join(rect_pth, "1")
        os.mkdir(data_pth)
        repeating_rectangles(data_pth, True)

        data_pth = os.path.join(rect_pth, "0")
        os.mkdir(data_pth)
        repeating_rectangles(data_pth, False)
    else:
        raise Exception("Repeating rectangle folder already exists.")

    # repeating circle patterns
    print('Creating Circle Patterns')
    circ_pth = os.path.join(root, "circle_patterns")
    if not os.path.isdir(circ_pth):
        os.mkdir(circ_pth)

        data_pth = os.path.join(circ_pth, "1")
        os.mkdir(data_pth)
        repeating_circles(data_pth, True)

        data_pth = os.path.join(circ_pth, "0")
        os.mkdir(data_pth)
        repeating_circles(data_pth, False)
    else:
        raise Exception("Repeating circle folder already exists.")

    # symmetric shapes
    print('Creating Symmetric Shapes')
    sym_pth = os.path.join(root, "sym_shapes")
    if not os.path.isdir(sym_pth):
        os.mkdir(sym_pth)

        data_pth = os.path.join(sym_pth, "1")
        os.mkdir(data_pth)
        symmetric_shapes(data_pth, True)

        data_pth = os.path.join(sym_pth, "0")
        os.mkdir(data_pth)
        symmetric_shapes(data_pth, False)
    else:
        raise Exception("Sym. shapes folder already exists.")

def repeating_circles(root, uniform = True):
    # [radii, radius_incs, x_space_incs, y_space_incs]
    # [  0  ,     1      ,      2      ,     3       ]
    all_bags = [RADII, RADIUS_INCS, X_SPACE_INCS, Y_SPACE_INCS]
    init_bags = [RADII, [0], [INIT_SPACE], [0]]  
    joint_variation_config = [[1], [2], [3], [1, 2], [1, 3], [2, 3]]

    patterned_shapes(root, all_bags, init_bags, joint_variation_config, draw_circles, "circles", uniform)

def repeating_rectangles(root, uniform = True):
    #permute_rectangles_80_3_30_2_0_0_20_0_True
    # [lengths, widths, rotations, length_incs, width_incs, rotation_incs, x_space_incs, y_space_incs]
    # [   0   ,    1  ,      2   ,       3    ,     4     ,       5      ,        6    ,       7     ]
    all_bags = [RECT_LENGTHS, RECT_WIDTHS, RECT_ROTATIONS, RECT_LENGTH_INCS, 
                RECT_WIDTH_INCS, RECT_ROTATION_INCS, RECT_X_SPACE_INCS, RECT_Y_SPACE_INCS]
    init_bags = [RECT_LENGTHS, RECT_WIDTHS, RECT_ROTATIONS, [0], [0], [0], [INIT_SPACE], [0]]
    joint_variation_config = [[3], [4], [6], [7], [3, 4], [3, 6], [3, 7], [4, 6], [4, 7]]
    # rot_inc, rot_inc + l_inc, rot_inc + w_inc mirrors random creation

    patterned_shapes(root, all_bags, init_bags, joint_variation_config, draw_rectangles, "rectangles", uniform)

def patterned_shapes(root, all_bags, init_bags, joint_variation_config, draw_func, shape, uniform):

    # create basic repeating shape
    permutations = generate_permutations(init_bags) 

    for config in joint_variation_config:
        bags = copy.deepcopy(init_bags) # copies ind. config (inner list)

        for var in config:
            bags[var] = all_bags[var]
        permutations += generate_permutations(bags)

    for i, perm in enumerate(permutations):
        img = Image.new('RGB', (DIM, DIM), color = 'white')
        for y in range(perm[1], DIM - perm[1], max(perm[1] * 5, int(DIM/5))):
            x = perm[0]/2
            img = draw_func(x, y, *perm, uniform, img)
        img.save(os.path.join(root, f"permute_{shape}_{'_'.join(list(map(str, perm)))}_{uniform}.jpg"), "JPEG")

def symmetric_shapes(root, uniform = True):
    for radius in SYM_RADII:
        for num_sections in SYM_NUM_SECTIONS:
            config_used, img = draw_symmetric_shape(num_sections, radius, uniform)
            img.save(os.path.join(root, f"symmetric_{radius}_{'_'.join(list(map(str, config_used)))}.jpg"), "JPEG")

########### DRAWING HELPERS ###########

def draw_rectangles(init_x, init_y, init_length, init_width, init_rotation, \
                    length_inc, width_inc, rotation_inc, x_space_inc, y_space_inc, uniform = True, img=None):
    leng, wid, rot, x, y = init_length, init_width, init_rotation, init_x, init_y

    #img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    while x < DIM and y < DIM:

        if not uniform:
            angle = np.radians(rot + np.random.uniform(30, 60))
            # l, w = leng + np.random.uniform(leng / 4, leng / 3), wid + np.random.uniform(wid / 4, wid / 3) # for more subtle changes
            l, w = leng + np.random.uniform(- leng / 3, leng / 3), wid + np.random.uniform(-wid / 3, wid / 3)
        else:
            angle = np.radians(rot)
            l, w = leng, wid

        c_angle, s_angle = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c_angle, -s_angle], [s_angle, c_angle]])
        rot_vertices = np.matmul(rot_matrix, np.array([[l/2, l/2, -l/2, -l/2], 
                                                       [w/2, -w/2, -w/2, w/2]]))

        vertices = rot_vertices + np.array([[x], [y]])
        draw.polygon(list(map(tuple, vertices.transpose())), fill = 'black')

        x += np.sqrt((leng/2) ** 2 + (wid/2) ** 2) + x_space_inc
        y += y_space_inc
        leng += length_inc
        wid += width_inc
        rot += rotation_inc
        
    return img

def draw_circles(init_x, init_y, init_radius, radius_inc, x_space_inc, y_space_inc, uniform = True, img=None):
    radius, x, y = init_radius, init_x, init_y

    #img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    while x < DIM and y < DIM:

        if not uniform:
            # r = radius + np.random.uniform(radius / 4, radius / 3) # for more subtle changes
            r = radius + np.random.uniform(- radius / 3, radius / 3)
        else:
            r = radius

        rel_vertices = np.array([[-r/2, r/2], 
                                 [-r/2, r/2]])

        vertices = rel_vertices + np.array([[x], [y]])
        draw.ellipse(list(map(tuple, vertices.transpose())), fill = 'black')

        x += r/2 + x_space_inc
        y += y_space_inc
        radius += radius_inc
        
    return img

def draw_symmetric_shape(num_sections, radius, uniform):
    img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    x_vertices = [radius]
    y_vertices = [0]

    ang_inc = np.pi / num_sections
    overall_config = []
    for i in range(num_sections):
        ang = (i + 1) * ang_inc
        end_x, end_y = np.cos(ang) * radius, np.sin(ang) * radius

        config =  np.random.randint(1, 5)
        overall_config.append(config)

        if config == 1:
            x_vertices.append(end_x)
            y_vertices.append(end_y)
        elif config == 2 or config == 3:
            height = np.random.uniform(radius / 8, radius / 4)
            mid_ang = ang - (ang_inc / 2)

            if config == 2:
                mid_x, mid_y = np.cos(mid_ang) * (radius + height), np.sin(mid_ang) * (radius + height)
            else:
                mid_x, mid_y = np.cos(mid_ang) * (radius - height), np.sin(mid_ang) * (radius - height)

            x_vertices += [mid_x, end_x]
            y_vertices += [mid_y, end_y]
        elif config == 4:
            circ_radius = np.sin(ang_inc / 2) * radius
            mid_ang = ang - (ang_inc / 2)
            cent_x, cent_y = np.cos(mid_ang) * radius, np.sin(mid_ang) * radius

            rel_vertices = np.array([[-circ_radius, circ_radius], 
                                     [-circ_radius, circ_radius]])
            vertices = rel_vertices + np.array([[cent_x + (DIM/2)], 
                                                [(DIM/2) + cent_y]])
            draw.ellipse(list(map(tuple, vertices.transpose())), fill = 'black')

            if uniform:
                reflected_vertices = rel_vertices + np.array([[cent_x + (DIM/2)], 
                                                              [(DIM/2) - cent_y]])
                draw.ellipse(list(map(tuple, reflected_vertices.transpose())), fill = 'black')
            else:
                reflected_vertices = (rel_vertices * np.array([[1] * 2, np.random.uniform(-1, 0, 2)])) + np.array([[cent_x + (DIM/2)], 
                                                                                                                   [(DIM/2) - cent_y]])
                draw.ellipse(list(map(tuple, reflected_vertices.transpose())), fill = 'black')

            x_vertices.append(end_x)
            y_vertices.append(end_y)

    rel_vertices = np.array([x_vertices, y_vertices])
    img_cent = np.array([[DIM/2], [DIM/2]]) 

    vertices = rel_vertices + img_cent
    draw.polygon(list(map(tuple, vertices.transpose())), fill = 'black')

    if uniform:
        reflected_verticies = (rel_vertices * np.array([[1], [-1]])) + img_cent 
        draw.polygon(list(map(tuple, reflected_verticies.transpose())), fill = 'black')
    else:
        reflected_verticies = (rel_vertices * np.array([[1] * len(x_vertices), np.random.uniform(-1, 0, len(y_vertices))])) + img_cent 
        draw.polygon(list(map(tuple, reflected_verticies.transpose())), fill = 'black')

    return overall_config, img

########### COLOUR HELPERS ###########

def recolor_image(image):
    colour_thresh = 127
    noise_mean = 0
    noise_std = 1

    colour_choices = [[102, 178, 255], [153, 153, 255], [255, 153, 51], [255, 102, 102], [51, 255, 153], [204, 0, 102]]
    colour = colour_choices[np.random.choice(range(len(colour_choices)), 1)[0]]          

    img_arr = np.array(np.asarray(image))

    for col_idx in range(len(colour)):
        thresh = np.where(img_arr[:,:,col_idx] < colour_thresh)
        img_arr[thresh[0],thresh[1],col_idx] = colour[col_idx]

    img_noise = Image.fromarray(img_arr + np.random.normal(noise_mean, noise_std, img_arr.shape).astype(np.uint8))
    img_noise = np.array(np.asarray(img_noise.filter(ImageFilter.GaussianBlur(radius = 3))))
    # ASSUMES SAME VALUES ACROSS ALL CHANNELS
    for i in range(3):
        img_noise[thresh[0],thresh[1], i] = img_arr[thresh[0],thresh[1], i] 

    #img_arr += np.random.normal(noise_mean, noise_std, img_arr.shape).astype(np.uint8)
    #img_arr = np.clip(img_arr, 0, 255) 
    img_arr = np.clip(img_noise, 0, 255)

    return Image.fromarray(img_arr)

########### MISC. HELPERS ###########

def generate_permutations(bags):
    if len(bags) == 1:
        return [[i] for i in bags[0]]

    partial_perms = generate_permutations(bags[1:])
    perms = []
    for i in bags[0]:
        perms += [[i] + p for p in partial_perms]

    return perms

# TODO:
# Use other shapes --> eg: triangle
# Try alternating different shapes

#folders = ["train", "val"]
#classes = ["0", "1"]

#for folder in folders:
#    for c in classes:
#        pth = os.path.join("Data", folder, c)
#        img_fnames = os.listdir(pth)

#        for fname in img_fnames:
#            img_pth = os.path.join(pth, fname)
#            img = Image.open(img_pth).convert('RGB')
#            img = recolor_image(img)
#            img.save(img_pth)