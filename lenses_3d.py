import cv2
import numpy as np


class Base3DLens:

    def __init__(self):
        pass

    def overlay(self, face_img, landmarks_map):
        position, rotation, scale = self._estimate_3d(landmarks_map)
        rendered_img = self._render(position, rotation, scale)
        return self._combine_3d_2d(face_img, rendered_img)

    def _estimate_3d(self, landmarks_map):
        # TODO
        pass

    def _render(self, position, rotation, scale):
        # TODO
        pass

    def _combine_3d_2d(self, face_img, rendered_img):
        # TODO
        return face_img


