import cv2
import time
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import *

from estimator_3d import Estimator3D


CHROMAKEY = (0, 0.69, 0.25)


class Panda3dApp(ShowBase):

    def _line(self, x, y, z, color):
        lines = LineSegs()
        lines.setColor(color)
        lines.moveTo(0, 0, 0)
        lines.drawTo(x, y, z)
        lines.setThickness(5)
        node = lines.create()
        np = NodePath(node)
        np.reparentTo(self.render)

    def _coords(self):
        self._line(5, 0, 0, (0, 0, 1, 1))
        self._line(0, 5, 0, (0, 1, 0, 1))
        self._line(0, 0, 5, (1, 0, 0, 1))

    def __init__(self):
        ShowBase.__init__(self)

        self.obj = self.loader.loadModel('data/cap.3ds')
        self.obj.reparentTo(self.render)
        self.obj.setPos(-1, 0, 1)
        self.obj.setScale(0.01)
        self.obj.setP(90)
        self.obj.setH(90)

        self.head = self.loader.loadModel('data/head.obj')
        self.head.setColor(*CHROMAKEY)
        self.head.reparentTo(self.render)
        self.head.setPos(-4.1, 0, -7.5)
        self.head.setScale(0.2)
        self.head.setH(90)

        # self.teapot = self.loader.loadModel('teapot')
        # self.teapot.reparentTo(self.render)
        # self.teapot.setPos(-1, 0, 1)

        self.cam.reparentTo(self.render)
        self.cam.setPos(10, 0, 0)
        self.cam.lookAt(0, 0, 0)
        self.camLens.setFov(50)

        self.setBackgroundColor(*CHROMAKEY)

        # self._coords()

        light = PointLight('plight')
        light.setColor((0.8, 0.8, 0.8, 1))
        light_np = self.render.attachNewNode(light)
        light_np.setPos(5, 0, 2)
        self.obj.setLight(light_np)

        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight.upcastToPandaNode())
        self.obj.setLight(alnp)


class Base3DLens:

    def __init__(self):
        # loadPrcFileData("", "window-type offscreen" ) # Spawn an offscreen buffer
        self.panda3d_app = Panda3dApp()
        self._estimator_3d = None

    def overlay(self, face_img, landmarks_map):
        self._estimator_3d = Estimator3D(
            landmarks_map, face_img.shape[1], face_img.shape[0]
        )

        # debug purposes
        rendered_img = landmarks_map.debug_draw_3d(self._estimator_3d, face_img)

        self._render()
        return rendered_img

    def _screenshot(self):
        # get getScreenshot as Texture, could be sped up using texture buffer?
        dr = self.panda3d_app.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()

        # Texture to opencv
        bytes_str = tex.getRamImageAs("RGB").getData()
        img = np.fromstring(bytes_str, np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
        img = cv2.flip(img, 0)  # dunno why but it is inverted in texture

        cv2.imshow('3d', img)
        if cv2.waitKey(1):
            pass

    def _render(self):
        start_t = time.time()

        self.panda3d_app.cam.setPos(*self._estimator_3d.get_cam_pos())
        self.panda3d_app.cam.lookAt(*self._estimator_3d.get_cam_target_pos())
        self.panda3d_app.cam.setR(self._estimator_3d.get_cam_roll() - 90)

        self.panda3d_app.graphicsEngine.renderFrame()

        print('Rendered frame in {} sec'.format(time.time() - start_t))


    def _combine_3d_2d(self, face_img, rendered_img):
        # TODO
        return face_img
