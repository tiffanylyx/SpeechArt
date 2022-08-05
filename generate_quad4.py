# This version constructs frameworks for every input sentence and generate structrue.

# You can find the source code of the following functions on this page. Some might not be avaliable on this page.
# https://docs.panda3d.org/1.10/python/_modules/index
from direct.showbase.ShowBase import ShowBase,LVecBase3,LQuaternion
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task


# You can find the source code of the following functions on this page. You can search the keyword on the left-up cornor.
# https://github.com/panda3d/panda3d/tree/dd3510eea743702400fe9aeb359d47bd2f5914ed/panda/src
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

# There are other python libraries
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
import os

from utils import *


if not os.path.exists('./texture'):
    os.makedirs('./texture')

# change the window size
loadPrcFileData('', 'win-size 1400 700')

# set up basic functions
base = ShowBase()
base.setBackgroundColor(0.7,0.7,0.7)
base.camLens.setFocalLength(1)
lens = PerspectiveLens()
base.cam.node().setLens(lens)
base.camLens.setFov(5,5)
base.camLens.setAspectRatio(2)
base.camLens.setFar(100)

# get the running time
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
    # store the generated Geom function
    square = Geom(vdata)
    square.addPrimitive(tris)
    return square

# load sentence vector model, mapping a sentence to a 100-dimensional vector
model_sentence = Doc2Vec.load("model/doc2vec100_text8.model")

# add lighting
'''
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

directionalLight = DirectionalLight('directionalLight')
directionalLight.setColor((0.8, 0.8, 0.8, 1))
directionalLightNP = render.attachNewNode(directionalLight)
# This light is facing forwards, away from the camera.
directionalLightNP.setHpr(0, -20, 0)
render.setLight(directionalLightNP)

directionalLight = DirectionalLight('directionalLight')
directionalLight.setColor((0.8, 0.8, 0.8, 1))
directionalLight.setShadowCaster(True, 512, 512)
directionalLightNP = render.attachNewNode(directionalLight)
# This light is facing forwards, away from the camera.
directionalLightNP.setHpr(20, 0, 0)
render.setLight(directionalLightNP)
'''
plight = PointLight('plight')
plight.setColor((0.8, 0.8, 0.8, 1))
plight.setShadowCaster(True, 512, 512)
plnp = render.attachNewNode(plight)
plnp.setPos(0, 0, 10)
render.setLight(plnp)
'''

render.setShaderAuto()

get_input = True
#base.disableMouse()
# set camera control
def spinCameraTask(task):
    base.camera.setZ(5)
    base.camera.lookAt(x_origin, y_origin,-1)
    #print(base.camLens.getFov())

    #base.camLens.setNear(y_origin+150)
    #base.camLens.setFar(y_center_r+400)

    return Task.cont

base.taskMgr.add(spinCameraTask, "SpinCameraTask")


while True:
    if get_input == True:
        # get input sentence
        s = input("Please enter a sentence:")
        sentence = s.lower()
        # stop sentence input
        get_input = False

        # compute the time difference between two sentence input
        now_time = time.time()
        time_period = now_time-previous_time

        # clean the sentence
        if sentence[-1]==".":
            sentence = sentence[:-1]

        # compute the 3D sentence vector of the input sentence
        sent_vect = compute_sent_vec(sentence, model_sentence)

        # add some randomness
        i0 = random.choice((1,-1))#test_positive(sent_vect[0])
        i1 = random.choice((1,-1))#test_positive(sent_vect[1])
        i2 = random.choice((1,-1))#test_positive(sent_vect[2])

        # compute the starting position of the new structure
        [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, time_period*0.5, i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2])
        z_origin = -3

        # compute parts of the speech of the given sentence
        res_parts = compute_sent_parts(sentence)

        # seperate the sentence into word list
        word_list = sentence.split(" ")

        # get the grammar analysis result. Split the sentence into :Noun Prase (NP) and Verb Prase(VP) and get the level of each word
        word_parts, res_key = get_cfg_structure(word_list)

        # compute the sentiment value of the given sentence
        sentiment = compute_sent_sentiment(sentence)

        # initiate variables
        x_center = 0
        y_center = 0
        z_center = 0
        points = []

        # clean the data
        split_word = word_parts[-1][0]
        sub_sentences = sentence.split(split_word)
        sub_sentences[0] = sub_sentences[0][:-1]
        sub_sentences[-1] = split_word+ sub_sentences[-1]
        print(word_parts,sub_sentences)

        count = 0

        # process the subsentence (NP and VP)
        for sub_sentence in sub_sentences:
            # compute the sentence vector of the sub sentence
            sub_sent_vect = compute_sent_vec(sub_sentence, model_sentence)
            sub_word_list = sub_sentence.split(' ')
            # compute the starting point based on the origin position, the length of the sub sentence and the sub sentence vector.
            [x_old_1, y_old_1,z_old_1] = solve_point_on_vector(x_origin, y_origin,z_origin, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])

            # process each word in the sub sentence
            for word in sub_word_list:
                if len(word)==0:
                    continue
                # add some randomness
                random.shuffle(sub_sent_vect)
                # compute the 3D word vector
                [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
                # compute the word length
                w = int(max(3,1.5*compute_word_length(word)))
                # add some +/- variance
                i0 = test_positive(nx)
                i1 = test_positive(ny)
                i2 = test_positive(nz)

                # compute the front surface of the framework
                # compute the second point of the framework
                [x2, y2, z2] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sub_sent_vect[0],i1*sub_sent_vect[1], i2*sub_sent_vect[2])

                x1 = x_old_1
                y1 = y_old_1
                z1 = z_old_1

                # use the word-level to determine the framework's height
                distance = 4*max(res_key.get(word, [0.5]))
                # add some random offset
                z_offset = distance*(0.4*random.random()-0.2)
                print("z_offset",z_offset)

                # compute the coordinates of the framework
                z1 = z1+z_offset
                z2 = z2+z_offset

                x3 = x1
                y3 = y1
                z3 = z1+distance

                x4 = x2
                y4 = y2
                z4 = z2+distance

                # the second point of this framework is used as the first point of the next framework
                x_old_1 = x2
                y_old_1 = y2
                z_old_1 = z2

                # compute the back surface of the framework
                # compute the first point's coordinate based on the previous result
                [x_move1, y_move1] = solve_moving_line(x1, y1, x2, y2, distance)

                # compute the second point's coordinate based on the previous result
                x_move2 = x_move1+x2-x1
                y_move2 = y_move1+y2-y1

                # add the framework structrue
                frame1 = base.loader.loadModel("models/box")
                frame1.setPosHprScale(LVecBase3(x1,y1,z1),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame1.setShaderAuto()
                frame1.reparentTo(render)

                frame2 = base.loader.loadModel("models/box")
                frame2.setPosHprScale(LVecBase3(x2,y2,z2),LVecBase3(0,0,0),LVecBase3(0.1, 0.1, distance))
                frame2.setShaderAuto()
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

                # compute the color value (3D word vector)
                color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)
                # if the word is a verb
                if res_parts[count][1] == 'VERB':
                    H = 0.4*abs(color_value[0])
                    print("verb")

                # if the word is noun
                elif res_parts[count][1] == 'NOUN':
                    print("Noun")
                    H = 0.6+0.4*abs(color_value[0])

                # if the word is other type
                else:
                    print("else")
                    H = 0.4+0.2*abs(color_value[0])

                # convert the HSV value to RGB value
                test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)

                # if the word is a noun, draw the vertical surfaces of the frame
                if res_parts[count][1] == 'NOUN':

                    # draw the front surface
                    square = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, [1,1,1,1])
                    # store the surface
                    snode = GeomNode('square')
                    snode.addGeom(square)
                    # compute the size of the surface to generate texture
                    width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    height = distance
                    # generate texture that has the word on it with the correct color
                    draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                    testTexture = loader.loadTexture("texture/"+word+".png")

                    node = NodePath(snode)
                    # set the alpha channel based on the word vector
                    node.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node.setColorScale(1,1,1,1)
                    node.setTwoSided(True)
                    node.setTexture(testTexture)
                    node.setShaderAuto()
                    #alpha_list.append(howOpaque)
                    node.reparentTo(render)
                    # compute the back surface of the framework
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
                    node.reparentTo(render)

                # if the word is a verb, draw the horizontal surfaces of the frame
                elif res_parts[count][1] == 'VERB':
                # Horizontal
                    width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    height = sqrt((x_move1-x1)**2 + (y_move1-y1)**2 + (z3-z1)**2)

                    draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                    testTexture = loader.loadTexture("texture/"+word+".png")

                    # draw the bottom surface of the framework
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
                    node.reparentTo(render)

                    # draw the top surface of the framework
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
                    node.reparentTo(render)


                count+=1

                # store the 8 points of the framework
                points = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
                [x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]]

                x_c = 0
                y_c = 0
                z_c = 0

                # compute the center of the framework
                for i in points:
                    x_c +=i[0]
                    y_c +=i[1]
                    z_c +=i[2]
                x_c_r = x_c/8
                y_c_r = y_c/8
                z_c_r = z_c/8

                # add some random points inside the framework
                for i in range(w):
                    p1 = random.choice([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])
                    p2 = random.choice([[x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]])

                    p_select = choice_random_point_on_line(p1, p2)

                    points.append(p_select)
                '''
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
                '''

                snode = GeomNode('square')

                # choose w surface
                for i in range(w):
                    # choose four points from the point-list to create the surface
                    [p_1, p_2, p_3, p_4] = random.choices(points, k=4)
                    p1 = copy.deepcopy(p_1)
                    p2 = copy.deepcopy(p_2)
                    p3 = copy.deepcopy(p_3)
                    p4 = copy.deepcopy(p_4)

                    # add some random offset
                    factor = 2

                    p1[0] = p_1[0]+factor*random.random()-factor/2
                    p1[1] = p_1[1]+factor*random.random()-factor/2

                    p2[0] = p_2[0]+factor*random.random()-factor/2
                    p2[1] = p_2[1]+factor*random.random()-factor/2

                    p3[0] = p_3[0]+factor*random.random()-factor/2
                    p3[1] = p_3[1]+factor*random.random()-factor/2

                    p4[0] = p_4[0]+factor*random.random()-factor/2
                    p4[1] = p_4[1]+factor*random.random()-factor/2

                    # draw the surface based on the computed color
                    square = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],test_color)
                    snode.addGeom(square)

                node = NodePath(snode)
                # set the alpha channel based on the word vector
                node.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node.setColorScale(1,1,1,0.9)
                node.setTwoSided(True)
                #alpha_list.append(howOpaque)
                node.reparentTo(render)


                if count==len(word_list):
                    print("This sentence is finished. Please press ctrl+c to start a new round")
                    get_input = True
                base.run()
