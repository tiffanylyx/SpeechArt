from direct.showbase.ShowBase import ShowBase,LVecBase3
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task

from panda3d.core import lookAt
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

import sys
import os
from math import pi, sin, cos, atan
import numpy as np
from sympy import *
import random

from utils import *

loadPrcFileData('', 'win-size 1200 600')

base = ShowBase()

# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec

# helper function to make a square given the Lower-Left-Hand and
# Upper-Right-Hand corners
def makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, color_value):

    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('square', format, Geom.UHStatic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    vertex.addData3(x1, y1, z1)
    vertex.addData3(x2, y2, z2)
    vertex.addData3(x3, y3, z3)
    vertex.addData3(x4, y4, z4)

    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))
    normal.addData3(normalized(0,0,1))

    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))
    color.addData4(abs(color_value[0]), abs(color_value[1]), abs(color_value[2]), abs(color_value[3]))

    #texcoord.addData2f(0.0, 1.0)
    #texcoord.addData2f(0.0, 0.0)
    #texcoord.addData2f(1.0, 0.0)
    #texcoord.addData2f(1.0, 1.0)
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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model_sentence = Doc2Vec.load("model/doc2vec100_text8.model")


x_old_1 = 0 # start time
y_old_1 = -10
z_old_1 = 0

snode = GeomNode('square')

alight = AmbientLight('alight')
alight.setColor((0.9, 0.9, 0.9, 0.4))
alnp = render.attachNewNode(alight)
render.setLight(alnp)
render.setAntialias(AntialiasAttrib.MAuto)

angleDegrees = 0
angleDegreesZ = 0
angleRadius = 30
angle1 = 0

#callback function to set  text
def setText(sentence):
    sentence = sentence.lower()
    global x_old_1,y_old_1,z_old_1snode,angle1
    textObject.setText(sentence)
    if sentence[-1]==".":
        sentence = sentence[:-1]
    x_old_1 = x_old_1+10
    y_old_1 = -10
    z_old_1 = 0


    sent_vect = compute_sent_vec(sentence, model_sentence)

    word_list = sentence.split(" ")

    res_parts = compute_sent_parts(sentence)
    count_2 = 0
    for word in word_list:
        random.shuffle(sent_vect)
        distance = 0.5 #word speak time
        w = compute_word_length(word)/2
        l = 2. # how to decide?
        i0 = random.choice((-1, 1))
        i1 = random.choice((-1, 1))
        i2 = random.choice((-1, 1))

        [x1, y1, z1] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sent_vect[0],i1*sent_vect[1], i2*sent_vect[2]) # spread along the sentence vec

        print(x1, y1, z1)
        angle1 = atan(y1/x1)*180
        print("angle1",angle1)
        if res_parts[count_2][1] == 'VERB':
            dim = 2
        else: dim = 3
        [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3) #word vector 2D or 3D

        [x2, y2, z2] = solve_point_on_vector(x1, y1, z1, w, nx, ny, nz)

        color_value = compute_word_vec(word, model, pca2, pca3, pca4, 4) #word vector 4D

        if count_2==0:
            x_old_1 = x1
            y_old_1 = y1
            z_old_1 = z1

            x_old_2 = x2
            y_old_2 = y2
            z_old_2 = z2

        else:
            square = makeQuad(x_old_1, y_old_1, z_old_1, x_old_2, y_old_2, z_old_2, x1, y1, z1, x2, y2, z2, color_value)
            x_old_1 = x1
            y_old_1 = y1
            z_old_1 = z1

            x_old_2 = x2
            y_old_2 = y2
            z_old_2 = z2

            snode.addGeom(square)
        count_2+=1

    cube = render.attachNewNode(snode)
    cube.setDepthWrite(True, 100)


    cube.setTwoSided(True)


    print("pos",base.camera.getHpr())

    print("pos",base.camera.getPos())

#clear the text
def clearText():
    entry.enterText('')
    
bk_text = "This is my Demo"
textObject = OnscreenText(text=bk_text, pos=(1.5, -0.95), scale=0.05,
                          fg=(1, 0.5, 0.5, 1), align=TextNode.ACenter,
                          mayChange=1)
#add text entry
entry = DirectEntry(text = "", scale=.05, command=setText,pos=(1.25, 0,-0.75),
initialText="Type Something", numLines = 2, focus=1, focusInCommand=clearText)



def spinCameraTask(task):
    global angle1, x_old_1,y_old_1,z_old_1
    angleDegrees = task.time * 6.0
    angleRadians = angleDegrees * (pi / 180.0)

    base.camera.setPos(0,y_old_1+10,0)
    base.camera.lookAt(x_old_1,y_old_1,z_old_1)
    return Task.cont


base.taskMgr.add(spinCameraTask, "SpinCameraTask")

#t = MyTapper()
base.run()
