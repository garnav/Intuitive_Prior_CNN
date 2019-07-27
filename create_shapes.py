# create_shapes.py
# Arnav Ghosh
# 27th July 2019

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# CONSTANTS
DIM = 512
INIT_SPACE = 20
LENGTH_BOUNDS= (5, 128)
WIDTH_BOUNDS = (10, 56)
ROTATION_BOUNDS = (0, 180)

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

        rel_vertices = np.matmul(rot_matrix, np.array([[-r/2, r/2], 
                                                       [r/2, -r/2]]))

        vertices = rel_vertices + np.array([[x], [y]])
        draw.ellipse(list(map(tuple, vertices.transpose())), fill = 'black')

        x += r/2 + x_space_inc
        y += y_space_inc
        radius += radius_inc
        
    return img

def patterned_circles(num_radius, num_r_incs, num_x_incs, num_y_incs, not_uniform = False):

    # TODO: Fix these
    # radii = 
    # radius_incs =
    # x_space_incs =
    # y_space_incs =

    # [radii, radius_incs, x_space_incs, y_space_incs]
    # [  0  ,     1      ,      2      ,     3       ]
    bags = [radii, radius_incs, x_space_incs, y_space_incs]
    init_bags = [radii, [0], [INIT_SPACE], [0]]  

    # repeating circles along x-axis
    permutations = generate_permutations(init_bags) 

    # vary variables
    joint_variation_config = [[1], [2], [3], [1, 2], [1, 3], [2, 3]]
    for config in joint_variation_config:
        bags = init_bags
        for var in config:
            bags[var] = all_bags[var]
            permutations += generate_permutations(bags)

    for i, perm in enumerate(permutations):
        x, y = perm[0]/2, DIM / 2

        img = draw_circles(x, y, *perm, not_uniform = not_uniform)
        img.save(f"permute_circle_{i}.jpg", "JPEG")

def patterned_rectangles(num_length, num_width, num_rot, num_l_incs, num_w_incs, num_r_incs, num_x_incs, num_y_incs, not_uniform = False):

    #TODO: Fix these
    # lengths = [LENGTH_BOUNDS[0]] + list(np.random.choice(range(*LENGTH_BOUNDS), num_length, replace = False))
    # widths = [WIDTH_BOUNDS[0]] + list(np.random.choice(range(*WIDTH_BOUNDS), num_width, replace = False))
    # rotations = [ROTATION_BOUNDS[0]] + list(np.random.choice(range(*ROTATION_BOUNDS), num_rot, replace = False))

    # length_incs = [0] + np.random.choice(range(LENGTH_BOUNDS[0], LENGTH_BOUNDS[0] * 2, int(LENGTH_BOUNDS[0] * np.random.uniform(1/LENGTH_BOUNDS[0], 0.2))), num_l_incs, replace = False)
    # width_incs = [0] + np.random.choice(range(WIDTH_BOUNDS[0], WIDTH_BOUNDS[0] * 2, int(WIDTH_BOUNDS[0] * np.random.uniform(1/WIDTH_BOUNDS[0], 0.2))), num_w_incs, replace = False)
    # rotation_incs = [0] + np.random.choice(range(30, 60, int(10 * np.random.uniform(0.1, 0.7))), num_r_incs, replace = False)
    # x_space_incs = [INIT_SPACE] + np.random.choice(range(INIT_SPACE, INIT_SPACE * 2, int(INIT_SPACE * np.random.uniform(1 / INIT_SPACE, 0.2))), num_x_incs, replace = False)
    # y_space_incs = [0] + np.random.choice(range(int(INIT_SPACE / 2), INIT_SPACE, int(INIT_SPACE * np.random.uniform(1 / INIT_SPACE, 0.1))), num_y_incs, replace = False)

    # [lengths, widths, rotations, length_incs, width_incs, rotation_incs, x_space_incs, y_space_incs]
    # [   0   ,    1  ,      2   ,       3    ,     4     ,       5      ,        6    ,       7     ]
    all_bags = [lengths, widths, rotations, length_incs, width_incs, rotation_incs, x_space_incs, y_space_incs]
    init_bags = [lengths, widths, rotations, [0], [0], [0], [INIT_SPACE], [0]]  

    # repeating rectangles along x-axis
    permutations = generate_permutations(init_bags) 

    # vary variables
    #joint_variation_config = [[3], [4], [5], [6], [7], [3, 4], [3, 5], [3, 6], [3, 7], [4, 5], [4, 6], [4, 7]]
    joint_variation_config = [[3, 6], [3, 7]]
    for config in joint_variation_config:
        bags = init_bags
        for var in config:
            bags[var] = all_bags[var]
            permutations += generate_permutations(bags)

    for i, perm in enumerate(permutations):
        x, y = perm[0]/2, DIM / 2

        img = draw_rectangles(x, y, *perm, not_uniform = not_uniform)
        img.save(f"permute_rectangle_{i}.jpg", "JPEG")
                     
def generate_permutations(bags):
    if len(bags) == 1:
        return [[i] for i in bags[0]]

    partial_perms = generate_permutations(bags[1:])
    perms = []
    for i in bags[0]:
        perms += [[i] + p for p in partial_perms]

    return perms



# circles init_x, init_y, init_radius, radius_inc, x_space_inc, y_space_inc
# triangles




# # length, width, rotation, x, y (latter determine spacing) 
# # each argument is a list indicating what happens on the 1st, 2nd third itertaion etc. wraps around
# # (all in step)
# def patterned_rectangles(length_patt, width_patt, rotation_patt, x_space_patt, y_space_patt, total_num):

#     img = Image.new('RGB', (DIM, DIM), color = 'white')
#     draw = ImageDraw.Draw(img)

#     leng, wid, rot = LENGTH_BOUNDS[0], WIDTH_BOUNDS[0], ROTATION_BOUNDS[0]
#     x_centers_dist = np.sqrt(((leng ** 2) + (wid ** 2))) + INIT_SPACE
#     y_centers_dist = 0
#     x, y = x_centers_dist/2, DIM/2

#     for j in range(total_num): 
#         i = 0
#         while x < DIM:
#             angle = np.radians(rot)
#             c_angle, s_angle = np.cos(angle), np.sin(angle)
#             rot_matrix = np.array([[c_angle, -s_angle], [s_angle, c_angle]])

#             rot_vertices = np.matmul(rot_matrix, np.array([[leng/2, leng/2, -leng/2, -leng/2], 
#                                                             [wid/2, -wid/2, -wid/2, wid/2]]))
#             vertices = rot_vertices + np.array([[x], [y]])
#             draw.polygon(list(map(tuple, vertices.transpose())), fill = 'black')

#             x +=  np.sqrt(((leng ** 2) + (wid ** 2))) + x_space_patt[i % len(x_space_patt)]
#             y += y_space_patt[i % len(y_space_patt)]
#             leng += length_patt[i % len(length_patt)]
#             wid += width_patt[i % len(width_patt)]
#             rot += rotation_patt[i % len(rotation_patt)]

#             i += 1

#         img.save(f"pattern_rectangle_{j}.jpg", "JPEG")

# def repeat_rectangles(num_length, num_width, num_rot, not_uniform = False):
#     lengths = [LENGTH_BOUNDS[0]] + list(np.random.choice(range(*LENGTH_BOUNDS), num_length, replace = False))
#     widths = [WIDTH_BOUNDS[0]] + list(np.random.choice(range(*WIDTH_BOUNDS), num_width, replace = False))
#     rotations = [ROTATION_BOUNDS[0]] + list(np.random.choice(range(*ROTATION_BOUNDS), num_rot, replace = False))

#     for rot in rotations:
#         for leng in lengths:
#             for wid in widths:
#                 x, y = leng/2, DIM / 2
#                 img = draw_rectangles(x, y, leng, wid, rot,  
#                                       length_inc=0, width_inc=0, 
#                                       rotation_inc=0, x_space_inc=INIT_SPACE, 
#                                       y_space_inc=0, not_uniform = not_uniform)
                
#                 img.save(f"repeat_rectangle_{leng}_{wid}_{rot}_{not not_uniform}.jpg", "JPEG")

# need main function that varies 1 at a time and then 2 etc.