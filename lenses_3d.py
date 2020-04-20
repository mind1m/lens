import cv2
import time
import numpy as np
import tempfile

from PIL import Image

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData, PNMImage
from panda3d.core import *

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        self.teapot = self.loader.loadModel('teapot')
        self.teapot.reparentTo(self.render)
        self.teapot.setPos(0, 0, 0)

        self.cam.reparentTo(self.render)
        self.cam.setPos(10, 0, 2)
        self.cam.lookAt(0, 0, 2)

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((1, 1, 1, 1))
        directionalLightNP = render.attachNewNode(directionalLight)
        directionalLightNP.setPos(20,0,2)
        directionalLightNP.lookAt(0,0,0)
        render.setLight(directionalLightNP)


class Base3DLens:

    def __init__(self):

        loadPrcFileData("", "window-type offscreen" ) # Spawn an offscreen buffer
        self.app = MyApp()

    def overlay(self, face_img, landmarks_map):
        position, rotation, scale = self._estimate_3d(landmarks_map)
        rendered_img = self._render(position, rotation, scale)
        return self._combine_3d_2d(face_img, rendered_img)

    def _estimate_3d(self, landmarks_map):
        # TODO
        return None, None, None

    def _render(self, position, rotation, scale):
        start_t = time.time()

        self.app.graphicsEngine.renderFrame()

        # get getScreenshot as Texture, could be sped up using texture buffer?
        dr = self.app.camNode.getDisplayRegion(0)
        tex = dr.getScreenshot()

        # Texture to opencv
        bytes_str = tex.getRamImageAs("RGB").getData()
        img = np.fromstring(bytes_str, np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
        img = cv2.flip(img, 0)  # dunno why but it is inverted in texture

        print('Rendered frame in {} sec'.format(time.time() - start_t))

        cv2.imshow('3d', img)
        if cv2.waitKey(1):
            pass

    def _combine_3d_2d(self, face_img, rendered_img):
        # TODO
        return face_img


