from panda3d.core import *
loadPrcFileData("", "textures-power-2 none")
loadPrcFileData("", "basic-shaders-only #t")
#loadPrcFileData("", "gl-version 3 2")
loadPrcFileData("", "notify-level-glgsg debug")


from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters

base = ShowBase()

plnp = NodePath('VolumetricLighting')
plnp.setPos(0, 25, 5)
base.cam.setPos(0, -28, 8)
pnda = loader.loadModel("panda")
pnda.reparentTo(render)

fltr = CommonFilters(base.win, base.cam)
fltr.setVolumetricLighting(plnp, 128, 5, 0.5, 1)

base.run()
