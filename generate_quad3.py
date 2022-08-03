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
from math import pi, sin, cos, atan, sqrt
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
def makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):

    format = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData('square', format, Geom.UHStatic)

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
plnp.setPos(0, 0, 0)
render.setLight(plnp)
plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(10, 0, 0)
render.setLight(plnp)
plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(0, 10, 0)
render.setLight(plnp)
plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(0, 0, 10)
render.setLight(plnp)
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
    base.camera.setZ(z_center_r+5)
    base.camera.lookAt(x_center_r,y_center_r,z_center_r)
    '''
    base.camLens.setNear(y_center_r+150)
    base.camLens.setFar(y_center_r+400)
    '''


    #base.camera.headsUp((x_center_r,y_center_r,z_center_r),(0,0,1))
    #base.camera.setP(0)
    #base.camera.setP(0)
    #base.camera.setR(0)
    #base.camera.setZ(z_center_r+4)

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
        #previous_time = now_time

        if sentence[-1]==".":
            sentence = sentence[:-1]

        # compute sentence vector

        sent_vect = compute_sent_vec(sentence, model_sentence)

        i0 = random.choice((1,-1))#test_positive(sent_vect[0])
        i1 = random.choice((1,-1))#test_positive(sent_vect[1])
        i2 = random.choice((1,-1))#test_positive(sent_vect[2])
        print(i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2])

        '''
        # compute the starting coordinate
        x_old_1 =  x_center_r + i0*min(6,time_period*0.1)
        y_old_1 = y_center_r + i1*min(6,time_period*0.1)
        z_old_1 = z_center_r + i2*min(6,time_period*0.1)
        '''

        [x_old_1, y_old_1,z_old_1] = solve_point_on_vector(0,0,0, time_period*0.08, i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2])


        # compute parts of the speech
        res_parts = compute_sent_parts(sentence)
        # seperate the sentence into word list
        word_list = sentence.split(" ")
        word_parts = get_cfg_structure(word_list)

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

        # generate points
        for word in word_list:
            random.shuffle(sent_vect)
            w = compute_word_length(word)/2
            [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
            i0 = test_positive(nx)
            i1 = test_positive(ny)
            i2 = test_positive(nz)

            [x1, y1, z1] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2]) # spread along the sentence vec
            [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
            [x2, y2, z2] = solve_point_on_vector(x1, y1, z1, w, nx, ny, nz)

            x_old_1 = x1
            y_old_1 = y1
            z_old_1 = z1
            x_c1 = x_c1+x1+x2
            y_c1 = y_c1+y1+y2
            z_c1 = z_c1+z1+z2
            points.append([x1, y1, z1] )
            points.append([x2, y2, z2] )

        # compute the center of all the points
        x_c1 = x_c1/(2*len(word_list))
        y_c1 = y_c1/(2*len(word_list))
        z_c1 = z_c1/(2*len(word_list))

        # compute the Voronoi diagram of the given points
        points = np.array(points)
        vor = Voronoi(points)
        Vertice = []
        r = vor.vertices
        # delete points that are far away from the center
        for i in r:
            if (abs(i[0]-x_c1)>1.5*len(word_list))|(abs(i[1]-y_c1)>1.5*len(word_list))|(abs(i[2]-z_c1)>1.5*len(word_list)):
                continue
            else:
                Vertice.append(i)

        for i in Vertice:
            x_center = x_center+i[0]
            y_center = y_center+i[1]
            z_center = z_center+i[2]

        x_center_r = x_center/len(Vertice)
        y_center_r = y_center/len(Vertice)
        z_center_r = z_center/len(Vertice)

        count = 0
        # generate surfaces based on the generated Voronoi vertices
        for word in word_list:
            w = compute_word_length(word)
            [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
            [p_1, p_2] = random.choices(Vertice, k=2)
            p1 = copy.deepcopy(p_1)
            p2 = copy.deepcopy(p_2)
            p1[0] = p1[0]+4*random.random()-2
            p1[1] = p1[1]+4*random.random()-2
            p1[2] = p1[2]+4*random.random()-2

            p2[0] = p2[0]+4*random.random()-2
            p2[1] = p2[1]+4*random.random()-2
            p2[2] = p2[2]+4*random.random()-2


            # compute the color based on the word vector
            color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)



            # if the word is verb, generate horizontal surface
            if res_parts[count][1] == 'VERB':
                H = 0.4*abs(color_value[0])
                print("verb")
                p1 = select_lowest_by_z(Vertice)
                p1[0] = p1[0]+4*random.random()-2
                p1[1] = p1[1]+4*random.random()-2
                p1[2] = p1[2]+4*random.random()-2
                low = p1[2]

                p2[2] = low
                [x3, y3, z3] = solve_point_on_vector(p1[0],p1[1],p1[2], w, nx, ny, nz)
                z3 = low
                x4 = x3+p2[0]-p1[0]
                y4 = y3+p2[1]-p1[1]
                z4 = z3+p2[2]-p1[2]
                #color_value = [0,0,0,0]

            # if the word is noun, generate vertical surface
            elif res_parts[count][1] == 'NOUN':
                print("Noun")
                H = 0.6+0.4*abs(color_value[0])
                p1_1 = select_lowest_by_y(Vertice)
                p1_2 = select_highest_by_y(Vertice)
                p1 = random.choice((p1_1, p1_2))
                p1[0] = p1[0]+4*random.random()-2
                p1[1] = p1[1]+4*random.random()-2
                p1[2] = p1[2]+4*random.random()-2

                value = p1[1]
                p2[1] = value
                [x3, y3, z3] = solve_point_on_vector(p1[0],p1[1],p1[2], w/2, nx, ny, nz)
                y3 = value
                x4 = x3+p2[0]-p1[0]
                y4 = y3+p2[1]-p1[1]
                z4 = z3+p2[2]-p1[2]

                #color_value = [1,1,1,1]

            else:
                print("else")
                H = 0.4+0.2*abs(color_value[0])
                [x3, y3, z3] = solve_point_on_vector(p1[0],p1[1],p1[2], w, nx, ny, nz)

                x4 = x3+p2[0]-p1[0]
                y4 = y3+p2[1]-p1[1]
                z4 = z3+p2[2]-p1[2]

            test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)


            x1 = (x_center_r +p1[0])/2
            print(p1[0], x1, x_center_r)
            y1 = (y_center_r +p1[1])/2
            z1 = (z_center_r +p1[2])/2

            x2 = (x_center_r +p2[0])/2
            y2 = (y_center_r +p2[1])/2
            z2 = (z_center_r +p2[2])/2

            x3 = (x_center_r +x3)/2
            y3 = (y_center_r +y3)/2
            z3 = (z_center_r +z3)/2

            x4 = (x_center_r +x4)/2
            y4 = (y_center_r +y4)/2
            z4 = (z_center_r +z4)/2

            # generate the surface
            square = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
            snode = GeomNode('square')
            snode.addGeom(square)
            count+=1

            width = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
            height = sqrt((x3-p1[0])**2 + (y3-p1[1])**2 + (z3-p1[2])**2)

            draw_text_texture(word, int(50*width), int(50*height), font, test_color)
            testTexture = loader.loadTexture("texture/"+word+".png")

            node = NodePath(snode)

            # set the alpha channel based on the word vector
            node.setTransparency(1)
            howOpaque=0.5+abs(color_value[2])*0.5
            node.setColorScale(1,1,1,howOpaque)
            node.setTwoSided(True)
            node.setTexture(testTexture)
            node_list.append(node)
            alpha_list.append(howOpaque)
            node.reparentTo(render)

            if count==len(word_list):
                print("This sentence is finished. Please press ctrl+c to start a new round")
                get_input = True
            base.run()
