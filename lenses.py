import cv2
import numpy as np


class BaseLens:

    FILENAME = None

    def __init__(self):
        self._img = cv2.imread(self.FILENAME, cv2.IMREAD_UNCHANGED)

    def img_copy(self):
        return self._img.copy()

    def overlay(self, face_img, landmarks_map):
        # use softened landmarks when available
        if not landmarks_map:
            return face_img

        lens_img = self.img_copy()

        return self._overlay(lens_img, face_img, landmarks_map)

    def _overlay(self, lens_img, face, landmarks_map):
        # should return image
        raise NotImplementedError

    def _angle_between(self, p1, p2):
        point = p2 - p1
        return -np.rad2deg(np.arctan2(point[1], point[0]))

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


class BaseTwoPointsLens(BaseLens):
    """
    Works for lenses when we want to align image by two landmarks.
    E.g. glasses can use ears.
    """
    LEFT_LANDMARK_KEY = None
    RIGHT_LANDMARK_KEY = None
    SCALE = 1
    OFFSET_X = 0  # from center between points, fraction of distances between points
    OFFSET_Y = 0  # from center between points, fraction of distances between points

    def _overlay(self, lens_img, face_img, landmarks_map):

        right_point = landmarks_map.get(self.RIGHT_LANDMARK_KEY)
        left_point = landmarks_map.get(self.LEFT_LANDMARK_KEY)
        between_points_dist = np.linalg.norm(left_point - right_point)

        # calculate how to resize
        img = self._resize_to_width(lens_img, between_points_dist * self.SCALE)

        angle = self._angle_between(left_point, right_point)
        angle_rad = np.deg2rad(angle)
        img = self._rotate(img, angle)

        center_x = int(left_point[0] + (right_point[0] - left_point[0]) / 2)
        center_y = int(left_point[1] + (right_point[1] - left_point[1]) / 2)

        base_offset_x = self.OFFSET_X * between_points_dist
        base_offset_y = self.OFFSET_Y * between_points_dist
        # now need to turn offsets to match points alignment
        offset_x = np.sin(angle_rad) * base_offset_x + np.sin(angle_rad) * base_offset_y
        offset_y = np.cos(angle_rad) * base_offset_x + np.cos(angle_rad) * base_offset_y

        res = self._blend(face_img, img, center_x + offset_x, center_y + offset_y)
        return res


class GlassesLens(BaseTwoPointsLens):

    FILENAME = 'data/glasses.png'
    LEFT_LANDMARK_KEY = 'left_ear'
    RIGHT_LANDMARK_KEY = 'right_ear'


class ClownNoseLens(BaseTwoPointsLens):

    FILENAME = 'data/clown_nose.png'
    LEFT_LANDMARK_KEY = 'nose_left'
    RIGHT_LANDMARK_KEY = 'nose_right'
    SCALE = 2


class LightningLens(BaseTwoPointsLens):

    FILENAME = 'data/lightning.png'
    LEFT_LANDMARK_KEY = 'left_eye_right'
    RIGHT_LANDMARK_KEY = 'right_eye_left'
    SCALE = 0.8
    OFFSET_Y = -1.3
