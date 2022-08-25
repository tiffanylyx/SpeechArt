from sympy import *

import numpy as np

import random

from math import sqrt

import time





# helper function to select the point with the lowest y
def select_lowest_by_y(points):
    y_min = 10000
    count = 0
    for point in points:
        if point[1]<y_min:
            min_index = count
            y_min = point[1]
        count += 1
    return points[min_index]

# helper function to select the point with the lowest y
def select_highest_by_y(points):
    y_max = -10000
    count = 0
    for point in points:
        if point[1]>y_max:
            max_index = count
            y_max = point[1]
        count += 1
    return points[max_index]

# helper function to select the point with the lowest z
def select_lowest_by_z(points):
    z_min = 10000
    count = 0
    for point in points:
        if point[2]<z_min:
            min_index = count
            z_min = point[2]
        count += 1
    return points[min_index]

def test_positive(num):
    if num>0:
        return 1
    elif num<0:
        return -1
    elif num==0:
        return 0

def keep_real(a):
    if not (a.is_real):
        a = abs(a)
    return a

def solve_start_position(x_old, y_old, z_old, distance, nx_s, ny_s, nz_s):
    time1 = time.time()

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    solved_value2=solve([x-x_old-distance,(x-x_old)/nx_s-(y-y_old)/ny_s,(x-x_old)/nx_s-(z-z_old)/nz_s], [x, y, z])

    r_x = keep_real(solved_value2[x])
    r_y = keep_real(solved_value2[y])
    r_z = keep_real(solved_value2[z])
    time2 = time.time()
    print("solve_start_position", time2-time1)

    return [r_x, r_y, r_z]
def solve_point_on_vector(x1, y1, z1, distance, vx, vy, vz):

    x2 = Symbol('x2')
    y2 = Symbol('y2')
    z2 = Symbol('z2')

    solved_value=solve([(x2-x1)**2+(y2-y1)**2+(z2-z1)**2-distance**2,(x2-x1)/vx-(y2-y1)/vy,(x2-x1)/vx-(z2-z1)/vz], [x2, y2, z2])


    r_x = keep_real(solved_value[0][0])
    r_y = keep_real(solved_value[0][1])
    r_z = keep_real(solved_value[0][2])

    return [r_x, r_y, r_z]

def solve_new_sec_vect(nx_s,ny_s, nz_s, x_old_1, y_old_1, z_old_1):
    vx = Symbol('vx')
    vy = Symbol('vy')

    solved_value=solve([vx*nx_s+vy*ny_s,vx*x_old_1+vy*y_old_1], [vx, vy])

    r_vx = keep_real(solved_value[vx])
    r_vy = keep_real(solved_value[vy])

    return [r_vx, r_vy, 0]

def solve_normal_with_three_point(p1,p2,p3):
    a = ( (p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]) );

    b = ( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]) );

    c = ( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]) );

    return [a,b,c]

def solve_quad(x1,y1,z1,nx,ny,nz,x2,w,l):
    y2 = Symbol('y2')
    z2 = Symbol('z2')
    solved_value2=solve([(x2-x1)**2+(y2-y1)**2+(z2-z1)**2-w**2,(x2-x1)*nx+(y2-y1)*ny+(z2-z1)*nz], [y2, z2])
    r_y2 = keep_real(solved_value2[0][0])
    r_z2 = keep_real(solved_value2[0][1])

    x3 = Symbol('x3')
    y3 = Symbol('y3')
    z3 = Symbol('z3')
    solved_value3=solve([(x3-x1)**2+(y3-y1)**2+(z3-z1)**2-l**2,
                         (x3-x1)*nx+(y3-y1)*ny+(z3-z1)*nz,
                         (x3-x1)*(x2-x1)+(y3-y1)*(r_y2-y1)+ (z3-z1)*(r_z2-z1)], [x3, y3, z3])

    r_x3 = keep_real(solved_value3[0][0])
    r_y3 = keep_real(solved_value3[0][1])
    r_z3 = keep_real(solved_value3[0][2])

    r_x4 = r_x3+(x2-x1)
    r_y4 = r_y3+(r_y2-y1)
    r_z4 = r_z3+(r_z2-z1)


    return [r_y2, r_z2, r_x3, r_y3, r_z3,r_x4, r_y4, r_z4]

def solve_moving_line(x1, y1, x2, y2, distance):
    a = Symbol('a')
    b = Symbol('b')
    solved_value =solve([(a-x1)**2+(b-y1)**2-distance**2,(a-x1)*(x1-x2)+(b-y1)*(y1-y2)], [a, b])

    pick = random.choice((0,1))
    r_a = keep_real(solved_value[pick][0])
    r_b = keep_real(solved_value[pick][1])

    return [r_a, r_b]



def choose_point_on_two_point(p1, p2, distance):

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')


    solved_value =solve([(x-p1[0])**2 + (y-p1[1])**2 +(z-p1[2])**2 - distance**2,(x-p1[0])*(p2[1]-p1[1])-(y-p1[1])*(p2[0]-p1[0]),(x-p1[0])*(p2[2]-p1[2])-(z-p1[2])*(p2[0]-p1[0])],[x,y,z])
    if ((keep_real(solved_value[0][0])-p2[0])**2 + (keep_real(solved_value[0][1])-p2[1])**2 +(keep_real(solved_value[0][2])-p2[2])**2>distance**2):
        r_x = keep_real(solved_value[0][0])
        r_y = keep_real(solved_value[0][1])
        r_z = keep_real(solved_value[0][2])
    elif ((keep_real(solved_value[1][0])-p2[0])**2 + (keep_real(solved_value[1][1])-p2[1])**2 +(keep_real(solved_value[1][2])-p2[2])**2>distance**2):
        r_x = keep_real(solved_value[1][0])
        r_y = keep_real(solved_value[1][1])
        r_z = keep_real(solved_value[1][2])



    return [r_x, r_y,r_z]


def choice_random_point_on_line(p1, p2):
    offset = random.random()

    r_x = p1[0]+offset*(p2[0]-p1[0])
    r_y = p1[1]+offset*(p2[1]-p1[1])
    r_z = p1[2]+offset*(p2[2]-p1[2])


    return [r_x, r_y,r_z]
