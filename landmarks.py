import cv2
import numpy as np


LANDMARKS_COUNT = 68


class LandmarksMap:

    # readable name tp dlib landmark index
    MAPPING = {
        'left_ear': 0,
        'right_ear': 16,
        'left_ear_bottom': 2,
        'right_ear_bottom': 14,
        'jaw_left': 4,
        'jaw_right': 12,
        'left_eye_left': 36,
        'left_eye_right': 39,
        'right_eye_left': 42,
        'right_eye_right': 45,
        'left_brow_center': 19,
        'right_brow_center': 24,
        'nose_top': 27,
        'nose_left': 31,
        'nose_right': 35,
        'nose_tip': 30,
        'nose_bottom': 33,
        'mouth_left': 48,
        'mouth_right': 64,
        'chin_left': 7,
        'chin_right': 9,
    }


    # 1 is face half-width
    COORDS_3D = {
        'nose_bottom': (0, 0, 0),  # origin
        'left_ear': (-1, 0.6, -1),
        'right_ear': (1, 0.6, -1),
        'left_ear_bottom': (-1, 0, -1),
        'right_ear_bottom': (1, 0, -1),
        'jaw_left': (-0.6, -0.45, -0.5),
        'jaw_right': (0.6, -0.45, -0.5),
        'chin_left': (-0.2, -0.9, 0),
        'chin_right': (0.2, -0.9, 0),
        'left_eye_left': (-0.6, 0.6, 0),
        'right_eye_right': (0.6, 0.6, 0),
    }

    @classmethod
    def from_dlib(cls, dlib_shape):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((LANDMARKS_COUNT, 2), dtype='int')
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, LANDMARKS_COUNT):
            coords[i] = np.array([dlib_shape.part(i).x, dlib_shape.part(i).y])
        # return the list of (x, y)-coordinates
        return cls(coords)

    def __init__(self, coords):
        self.coords = coords

    def get(self, name):
        return self.coords[self.MAPPING[name]]

    def get_3d(self, name):
        return self.COORDS_3D[name]

    def debug_draw(self, img):

        for i in range(LANDMARKS_COUNT):
            x, y = self.coords[i]
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(
                img, str(i), (x - 10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 100, 0), 1
            )

        for name in self.MAPPING.keys():
            x, y = self.get(name)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(
                img, name, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1
            )
        return img

    def debug_draw_3d(self, estimator_3d, img):
        for name in self.COORDS_3D.keys():
            x, y = self.get(name)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(
                img, name, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 0), 1
            )

        nose_line_start = self.get('nose_bottom')
        nose_x_axis = estimator_3d.project_to_2d(1, 0, 0)
        nose_y_axis = estimator_3d.project_to_2d(0, 1, 0)
        nose_z_axis = estimator_3d.project_to_2d(0, 0, 1)
        cv2.line(img, tuple(nose_line_start), tuple(nose_x_axis), (255, 0, 0), 2)
        cv2.putText(
            img, 'X', tuple(nose_x_axis),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 0, 0), 1
        )

        cv2.line(img, tuple(nose_line_start), tuple(nose_y_axis), (0, 255, 0), 2)
        cv2.putText(
            img, 'Y', tuple(nose_y_axis),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 0, 0), 1
        )

        cv2.line(img, tuple(nose_line_start), tuple(nose_z_axis), (0, 0, 255), 2)
        cv2.putText(
            img, 'Z', tuple(nose_z_axis),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 0, 0), 1
        )


def smooth_landmarks(landmarks_maps):
    # take landmarks_maps and produce smoothed LandmarksMap
    # to avoid wiggling

    rapid_movement = False

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((LANDMARKS_COUNT, 2), dtype='int')

    # iterate all landmarks
    for ld_idx in range(LANDMARKS_COUNT):
        # pre-fill zeros for all maps
        l_coords = np.zeros((len(landmarks_maps), 2), dtype='int')
        for map_idx, lm in enumerate(landmarks_maps):
            l_coords[map_idx, :] = lm.coords[ld_idx, :]

        # calculate max std
        std = np.max(np.std(l_coords, axis=0))
        if std < 7:
            # not much movement, just average all
            coords[ld_idx, :] = np.mean(l_coords, axis=0)
        else:
            # rapid movement, take last only
            rapid_movement = True
            coords[ld_idx, :] = l_coords[-1, :]

    return LandmarksMap(coords), rapid_movement
