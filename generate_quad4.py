# This version focuses more on parts of speech and word senti
from direct.showbase.ShowBase import ShowBase,LVecBase3,LQuaternion
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task

from panda3d.core import lookAt,AlphaTestAttrib,RenderAttrib
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter,GeomTristrips
from panda3d.core import Texture, GeomNode
from panda3d.core import PointLight, DirectionalLight, AmbientLight,AntialiasAttrib
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode
from panda3d.core import LVector3
from panda3d.core import loadPrcFileData
from panda3d.core import NodePath
from panda3d.core import Material
from panda3d.core import Lens
from panda3d.core import ColorAttrib,ColorBlendAttrib

import sys
import os
from math import pi, sin, cos, atan, sqrt, atan2
import numpy as np
from sympy import *
import random
import time
from scipy.spatial import Voronoi
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import copy
import colorsys

from utils import *

import os
if not os.path.exists('./texture'):
    os.makedirs('./texture')

# change the window size
loadPrcFileData('', 'win-size 1400 700')

base = ShowBase()
base.setBackgroundColor(0.7,0.7,0.7)
base.camLens.setFocalLength(30)


previous_time = time.time()
# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec

# generate a surface with given vertexs and color
def makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, color_value):

    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('square', format, Geom.UHStatic)
    color = GeomVertexWriter(vdata, 'color')
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    vertex.addData3(x1, y1, z1)
    vertex.addData3(x2, y2, z2)
    vertex.addData3(x3, y3, z3)
    vertex.addData3(x4, y4, z4)

    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))

    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), 1)
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), 1)
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), 1)
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), 1)

    texcoord.addData2f(0, 0)
    texcoord.addData2f(0, 1)
    texcoord.addData2f(1, 0)
    texcoord.addData2f(1, 1)

    tris = GeomTristrips(Geom.UHDynamic)
    tris.addVertices(0,1,2,3)
    '''
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(1, 2, 3)
    '''
    square = Geom(vdata)
    square.addPrimitive(tris)
    return square

# load sentence vector model
model_sentence = Doc2Vec.load("model/doc2vec100_text8.model")

# add lighting

alight = AmbientLight('alight')
alight.setColor((0.9, 0.9, 0.9, 0.4))
alnp = render.attachNewNode(alight)
render.setLight(alnp)
render.setAntialias(AntialiasAttrib.MAuto)

plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(0, 10, 10)
render.setLight(plnp)
'''
plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(0, 0, 10)
render.setLight(plnp)
'''
render.setShaderAuto()



x_center = 0
y_center = 0
z_center = 0



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


x_center_r = 0
y_center_r = 0
z_center_r = 0
get_input = True
#base.disableMouse()
# set camera control
def spinCameraTask(task):
    global x_center_r,y_center_r,z_center_r
    base.camera.setZ(5)
    base.camera.lookAt(x_origin, y_origin,z_origin)



    #base.camLens.setNear(y_origin+150)
    #base.camLens.setFar(y_center_r+400)


    return Task.cont
base.taskMgr.add(spinCameraTask, "SpinCameraTask")
def test_positive(num):
    if num>0:
        return 1
    elif num<0:
        return -1
    elif num==0:
        return 0
while True:
    if get_input == True:
        # get input sentence
        s = input("Please enter a sentence:")
        sentence = s.lower()
        get_input = False


        # compute the time difference between two sentence input
        now_time = time.time()
        time_period = now_time-previous_time
        print(time_period)
        #previous_time = now_time

        if sentence[-1]==".":
            sentence = sentence[:-1]

        # compute sentence vector

        sent_vect = compute_sent_vec(sentence, model_sentence)

        i0 = random.choice((1,-1))#test_positive(sent_vect[0])
        i1 = random.choice((1,-1))#test_positive(sent_vect[1])
        i2 = random.choice((1,-1))#test_positive(sent_vect[2])


        [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, time_period*0.5, i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2])
        z_origin = -3
        print("x_origin, y_origin,z_origin",x_origin, y_origin,z_origin)

        # compute parts of the speech
        res_parts = compute_sent_parts(sentence)
        print("res_parts",res_parts)
        # seperate the sentence into word list
        word_list = sentence.split(" ")

        word_parts, res_key = get_cfg_structure(word_list)

        sentiment = compute_sent_sentiment(sentence)

        # initiate variables
        x_center = 0
        y_center = 0
        z_center = 0
        points = []
        x_c1 = 0
        y_c1 = 0
        z_c1 = 0
        node_list = []
        alpha_list = []

        split_word = word_parts[-1][0]
        sub_sentences = sentence.split(split_word)
        sub_sentences[0] = sub_sentences[0][:-1]
        sub_sentences[-1] = split_word+ sub_sentences[-1]
        print(word_parts,sub_sentences)

        x_center = 0
        y_center = 0
        z_center = 0

        count = 0

        for sub_sentence in sub_sentences:
            sub_sent_vect = compute_sent_vec(sub_sentence, model_sentence)

            sub_word_list = sub_sentence.split(' ')
            [x_old_1, y_old_1,z_old_1] = solve_point_on_vector(x_origin, y_origin,z_origin, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])

            for word in sub_word_list:
                if len(word)==0:
                    continue
                random.shuffle(sub_sent_vect)
                [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
                w = int(max(3,1.5*compute_word_length(word)))
                i0 = test_positive(nx)
                i1 = test_positive(ny)
                i2 = test_positive(nz)

                [x2, y2, z2] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sub_sent_vect[0],i1*sub_sent_vect[1], i2*sub_sent_vect[2]) # spread along the sentence vec

                x1 = x_old_1
                y1 = y_old_1
                z1 = z_old_1


                distance = 4*max(res_key.get(word, [0.5]))

                z_offset = distance*(random.random()-0.5)
                print("z_offset",z_offset)

                z1 = z1+z_offset
                z2 = z2+z_offset

                x3 = x1
                y3 = y1
                z3 = z1+distance

                x4 = x2
                y4 = y2
                z4 = z2+distance


                x_old_1 = x2
                y_old_1 = y2
                z_old_1 = z2

                '''
                x_center = x_center+x1+x2+x3+x4
                y_center = y_center+y1+y2+y3+y4
                z_center = z_center+z1+z2+z3+z4
                '''

                [x_move1, y_move1] = solve_moving_line(x1, y1, x2, y2, distance)


                x_move2 = x_move1+x2-x1
                y_move2 = y_move1+y2-y1


                frame1 = base.loader.loadModel("models/box")
                frame1.setPosHprScale(LVecBase3(x1,y1,z1),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame1.reparentTo(render)

                frame2 = base.loader.loadModel("models/box")
                frame2.setPosHprScale(LVecBase3(x2,y2,z2),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame2.reparentTo(render)

                frame3 = base.loader.loadModel("models/box")
                frame3.setPosHprScale(LVecBase3(x_move1,y_move1,z1),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame3.reparentTo(render)

                frame4 = base.loader.loadModel("models/box")
                frame4.setPosHprScale(LVecBase3(x_move2,y_move2,z2),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame4.reparentTo(render)

                frame5 = base.loader.loadModel("models/box")
                frame5.setPosHprScale(LVecBase3(x1,y1,z1),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance, 0.1, 0.1))
                frame5.reparentTo(render)

                frame6 = base.loader.loadModel("models/box")
                frame6.setPosHprScale(LVecBase3(x2,y2,z2),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance, 0.1, 0.1))
                frame6.reparentTo(render)

                frame7 = base.loader.loadModel("models/box")
                frame7.setPosHprScale(LVecBase3(x1,y1,z3),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance, 0.1, 0.1))
                frame7.reparentTo(render)

                frame8 = base.loader.loadModel("models/box")
                frame8.setPosHprScale(LVecBase3(x2,y2,z4),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance, 0.1, 0.1))
                frame8.reparentTo(render)


                color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)
                if res_parts[count][1] == 'VERB':
                    H = 0.4*abs(color_value[0])
                    print("verb")

                # if the word is noun, generate vertical surface
                elif res_parts[count][1] == 'NOUN':
                    print("Noun")
                    H = 0.6+0.4*abs(color_value[0])

                else:
                    print("else")
                    H = 0.4+0.2*abs(color_value[0])

                test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)


                if res_parts[count][1] == 'NOUN':
                    # Vertical

                    square = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, [1,1,1,1])

                    snode = GeomNode('square')
                    snode.addGeom(square)

                    width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    height = distance

                    draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                    testTexture = loader.loadTexture("texture/"+word+".png")

                    node = NodePath(snode)
                    # set the alpha channel based on the word vector
                    node.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node.setColorScale(1,1,1,1)
                    node.setTwoSided(True)
                    node.setTexture(testTexture)
                    node_list.append(node)
                    #alpha_list.append(howOpaque)
                    node.reparentTo(render)

                    square = makeQuad(x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, [1,1,1,1])

                    snode = GeomNode('square')
                    snode.addGeom(square)

                    node = NodePath(snode)
                    # set the alpha channel based on the word vector
                    node.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node.setColorScale(1,1,1,1)
                    node.setTwoSided(True)
                    node.setTexture(testTexture)
                    node_list.append(node)
                    #alpha_list.append(howOpaque)
                    node.reparentTo(render)

                elif res_parts[count][1] == 'VERB':
                # Horizontal

                    width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    height = sqrt((x_move1-x1)**2 + (y_move1-y1)**2 + (z3-z1)**2)

                    draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                    testTexture = loader.loadTexture("texture/"+word+".png")

                    square = makeQuad(x1, y1, z1, x2, y2, z2, x_move1, y_move1, z1, x_move2, y_move2, z2, [1,1,1,1])
                    snode = GeomNode('square')
                    snode.addGeom(square)
                    node = NodePath(snode)

                    # set the alpha channel based on the word vector
                    node.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node.setColorScale(1,1,1,1)
                    node.setTwoSided(True)
                    node.setTexture(testTexture)
                    node_list.append(node)
                    node.reparentTo(render)

                    square = makeQuad(x1, y1, z3, x2, y2, z4, x_move1, y_move1, z3, x_move2, y_move2, z4, [1,1,1,1])
                    snode = GeomNode('square')
                    snode.addGeom(square)
                    node = NodePath(snode)

                    # set the alpha channel based on the word vector
                    node.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node.setColorScale(1,1,1,1)
                    node.setTwoSided(True)
                    node.setTexture(testTexture)
                    node_list.append(node)
                    node.reparentTo(render)


                count+=1
                '''
                x_center = x_center+2*(x_move1+x_move2)
                y_center = y_center+2*(y_move1+y_move2)
                z_center = z_center+z1+z2+z3+z4

                x_center = x_center/(8*count)
                y_center = y_center/(8*count)
                z_center = z_center/(8*count)
                '''

                points = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
                [x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]]

                x_c = 0
                y_c = 0
                z_c = 0

                for i in points:
                    x_c +=i[0]
                    y_c +=i[1]
                    z_c +=i[2]
                x_c_r = x_c/8
                y_c_r = y_c/8
                z_c_r = z_c/8


                for i in range(w):
                    p1 = random.choice([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])
                    p2 = random.choice([[x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]])

                    p_select = choice_random_point_on_line(p1, p2)

                    points.append(p_select)
                points = np.array(points)
                vor = Voronoi(points)

                Vertice = []
                r = vor.vertices

                # delete points that are far away from the center
                for i in r:
                    if (abs(i[0]-x_c_r)>1*w)|(abs(i[1]-y_c_r)>1*w)|(abs(i[2]-z_c_r)>1*w):
                        continue
                    else:
                        Vertice.append(i)

                Vertice.extend(points)

                snode = GeomNode('square')

                for i in range(w):
                    [p_1, p_2, p_3, p_4] = random.choices(points, k=4)
                    p1 = copy.deepcopy(p_1)
                    p2 = copy.deepcopy(p_2)
                    p3 = copy.deepcopy(p_3)
                    p4 = copy.deepcopy(p_4)

                    #p4 = [0,0,0]
                    factor = 2

                    p1[0] = p_1[0]+factor*random.random()-factor/2
                    p1[1] = p_1[1]+factor*random.random()-factor/2

                    p2[0] = p_2[0]+factor*random.random()-factor/2
                    p2[1] = p_2[1]+factor*random.random()-factor/2

                    p3[0] = p_3[0]+factor*random.random()-factor/2
                    p3[1] = p_3[1]+factor*random.random()-factor/2


                    p4[0] = p_4[0]+factor*random.random()-factor/2
                    p4[1] = p_4[1]+factor*random.random()-factor/2


                    #square = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],test_color)

                    #snode.addGeom(square)

                node = NodePath(snode)
                # set the alpha channel based on the word vector
                node.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node.setColorScale(1,1,1,0.9)
                node.setTwoSided(True)
                node_list.append(node)
                #alpha_list.append(howOpaque)
                node.reparentTo(render)


                if count==len(word_list):
                    print("This sentence is finished. Please press ctrl+c to start a new round")
                    get_input = True
                base.run()
