import cv2
import numpy as np


class BaseLens:

    def overlay(self, face):
        # should return image
        raise NotImplementedError

    def _angle_between(self, p1, p2):
        point = p2 - p1
        return np.rad2deg(np.arctan2(point[1], point[0]))

    def _rotate(self, img, angle):
        # before rotation, we need to pad image so it does not go out of border
        # we want it to become a square
        max_side = max(img.shape[0], img.shape[1])
        # how much we pad from each side (left/right)
        pad_horiz = int((max_side - img.shape[1]) / 2)
        # how much we pad from each side (top/bottom)
        pad_vert = int((max_side - img.shape[0]) / 2)
        color = [0, 0, 0, 0]  # transparent black
        img = cv2.copyMakeBorder(
            img,
            pad_vert, pad_vert, pad_horiz, pad_horiz,
            cv2.BORDER_CONSTANT, value=color
        )

        # now do the actual rotation
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            img, rot_mat, img.shape[1::-1],
            flags=cv2.INTER_LINEAR
        )
        return result

    def _blend(self, orig_img, lens_img, center_x, center_y):

        x1 = int(center_x - lens_img.shape[1] / 2)
        y1 = int(center_y - lens_img.shape[0] / 2)

        alpha_lens = lens_img[:, :, 3] / 255.0
        alpha_orig = 1.0 - alpha_lens

        x2 = x1 + lens_img.shape[1]
        y2 = y1 + lens_img.shape[0]

        for c in range(0, 3):
            orig_img[y1:y2, x1:x2, c] = (
                alpha_lens * lens_img[:, :, c] +
                alpha_orig * orig_img[y1:y2, x1:x2, c]
            )

        return orig_img


class GlassesLens(BaseLens):
    FILENAME = 'data/glasses.png'

    def __init__(self):
        self._img = cv2.imread(self.FILENAME, cv2.IMREAD_UNCHANGED)

    def overlay(self, face, face_img=None):
        if face_img is None:
            face_img = face._img.copy()

        img = self._img.copy()

        if not face.mapper:
            return face_img

        right_ear = face.mapper.get('right_ear')
        left_ear = face.mapper.get('left_ear')
        ears_width = np.linalg.norm(left_ear - right_ear)

        # calculate how to resize glasses
        scale = ears_width / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        # resize image
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        angle = self._angle_between(left_ear, right_ear)
        img = self._rotate(img, -angle)

        center_x = int(left_ear[0] + (right_ear[0] - left_ear[0]) / 2)
        center_y = int(left_ear[1] + (right_ear[1] - left_ear[1]) / 2)

        res = self._blend(face_img, img, center_x, center_y)
        return res

    def overlay_debug(self, face):
        img = self.overlay(face)
        return face.debug_draw(img=img)
