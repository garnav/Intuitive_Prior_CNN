# create_shapes.py
# Arnav Ghosh
# 27th July 2019

from shape_constants import *

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def draw_rectangles(init_x, init_y, init_length, init_width, init_rotation, \
                    length_inc, width_inc, rotation_inc, x_space_inc, y_space_inc, not_uniform = False):
    leng, wid, rot, x, y = init_length, init_width, init_rotation, init_x, init_y

    img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    while x < DIM and y < DIM:

        if not_uniform:
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

def draw_circles(init_x, init_y, init_radius, radius_inc, x_space_inc, y_space_inc, not_uniform = False):
    radius, x, y = init_radius, init_x, init_y

    img = Image.new('RGB', (DIM, DIM), color = 'white')
    draw = ImageDraw.Draw(img)

    while x < DIM and y < DIM:

        if not_uniform:
            # r = radius + np.random.uniform(radius / 4, radius / 3) # for more subtle changes
            r = radius + np.random.uniform(- radius / 3, radius / 3)
        else:
            r = radius

        rel_vertices = np.array([[-r/2, r/2], 
                                 [-r/2, r/2]])

        vertices = rel_vertices + np.array([[x], [y]])
        print(vertices)
        draw.ellipse(list(map(tuple, vertices.transpose())), fill = 'black')

        x += r/2 + x_space_inc
        y += y_space_inc
        radius += radius_inc
        
    return img

def patterned_circles(num_radius, num_r_incs, num_x_incs, num_y_incs, not_uniform = False):
    # [radii, radius_incs, x_space_incs, y_space_incs]
    # [  0  ,     1      ,      2      ,     3       ]
    all_bags = [RADII, RADIUS_INCS, X_SPACE_INCS, Y_SPACE_INCS]
    init_bags = [RADII, [0], [INIT_SPACE], [0]]  
    joint_variation_config = [[1], [2], [3], [1, 2], [1, 3], [2, 3]]

    patterned_shapes(all_bags, init_bags, joint_variation_config, draw_circles, "circles", not_uniform)

def patterned_rectangles(num_length, num_width, num_rot, num_l_incs, num_w_incs, num_r_incs, num_x_incs, num_y_incs, not_uniform = False):
    # [lengths, widths, rotations, length_incs, width_incs, rotation_incs, x_space_incs, y_space_incs]
    # [   0   ,    1  ,      2   ,       3    ,     4     ,       5      ,        6    ,       7     ]
    all_bags = [RECT_LENGTHS, RECT_WIDTHS, RECT_ROTATIONS, RECT_LENGTH_INCS, 
                RECT_WIDTH_INCS, RECT_ROTATION_INCS, RECT_X_SPACE_INCS, RECT_Y_SPACE_INCS]
    init_bags = [RECT_LENGTHS, RECT_WIDTHS, RECT_ROTATIONS, [0], [0], [0], [INIT_SPACE], [0]]
    joint_variation_config = [[3], [4], [6], [7], [3, 4], [3, 6], [3, 7], [4, 6], [4, 7]]
    #joint_variation_config = [[3, 6], [3, 7]]  

    # rot_inc, rot_inc + l_inc, rot_inc + w_inc looks weird (like things following over or domino like)

    patterned_shapes(all_bags, init_bags, joint_variation_config, draw_rectangles, "rectangles", not_uniform)

def patterned_shapes(all_bags, init_bags, joint_variation_config, draw_func, shape, not_uniform):

    # create basic repeating shape
    permutations = generate_permutations(init_bags) 

    for config in joint_variation_config:
        bags = init_bags
        for var in config:
            bags[var] = all_bags[var]
            permutations += generate_permutations(bags)

    for i, perm in enumerate(permutations):
        x, y = perm[0]/2, DIM / 2 #TODO: How do we know the first one is length

        img = draw_func(x, y, *perm, not_uniform)
        img.save(f"permute_{shape}_{'_'.join(list(map(str, perm)))}_{not not_uniform}.jpg", "JPEG")
                 
def generate_permutations(bags):
    if len(bags) == 1:
        return [[i] for i in bags[0]]

    partial_perms = generate_permutations(bags[1:])
    perms = []
    for i in bags[0]:
        perms += [[i] + p for p in partial_perms]

    return perms

def alternating_shapes():
    # global like x_inc, y_inc
    # shape specific info
    pass

def symmetric_shapes():
    pass


# triangles

# need main function that varies 1 at a time and then 2 etc.

# Symmetry
# To create symettry across a line, taking and existing image and reflect
# Shapes --> separate into quarters and decide what to do for each quarter and reflect across y=x, x, y

