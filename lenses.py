import cv2
import numpy as np


class BaseLens:

    FILENAME = None

    def __init__(self):
        self._img = cv2.imread(self.FILENAME, cv2.IMREAD_UNCHANGED)

    def img_copy(self):
        return self._img.copy()

    def overlay(self, face, smoothed_landmarks_map=None):
        # use softened landmarks when available
        landmarks_map = face.landmarks_map
        if smoothed_landmarks_map:
            landmarks_map = smoothed_landmarks_map

        face_img = face._img.copy()
        if not landmarks_map:
            return face_img

        lens_img = self.img_copy()

        return self._overlay(lens_img, face_img, landmarks_map)

    def _overlay(self, lens_img, face, landmarks_map):
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

    def _resize_to_width(self, img, to_width):
        # make img be width wide, keeping aspect ratio
        scale = to_width / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        # resize image
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class GlassesLens(BaseLens):

    FILENAME = 'data/glasses.png'

    def _overlay(self, lens_img, face_img, landmarks_map):

        right_ear = landmarks_map.get('right_ear')
        left_ear = landmarks_map.get('left_ear')
        ears_width = np.linalg.norm(left_ear - right_ear)

        # calculate how to resize glasses
        img = self._resize_to_width(lens_img, ears_width)

        angle = self._angle_between(left_ear, right_ear)
        img = self._rotate(img, -angle)

        center_x = int(left_ear[0] + (right_ear[0] - left_ear[0]) / 2)
        center_y = int(left_ear[1] + (right_ear[1] - left_ear[1]) / 2)

        res = self._blend(face_img, img, center_x, center_y)
        return res


class ClownNoseLens(BaseLens):

    FILENAME = 'data/clown_nose.png'

    def _overlay(self, lens_img, face_img, landmarks_map):

        nose_left = landmarks_map.get('nose_left')
        nose_right = landmarks_map.get('nose_right')
        nose_width = np.linalg.norm(nose_left - nose_right)

        img = self._resize_to_width(lens_img, nose_width * 2)

        angle = self._angle_between(nose_left, nose_right)
        img = self._rotate(img, -angle)

        center_x = int(nose_left[0] + (nose_right[0] - nose_left[0]) / 2)
        center_y = int(nose_left[1] + (nose_right[1] - nose_left[1]) / 2)

        res = self._blend(face_img, img, center_x, center_y)
        return res
