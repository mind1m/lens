import cv2
import time
import numpy as np

from scipy.spatial.transform import Rotation


class Estimator3D:

    def __init__(self, landmarks_map, image_width, image_height):
        self.rot = None
        self.trans = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # camera params
        focal_length = image_width
        center = (image_height / 2, image_width / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        # assemble 2d-3d correspondence points
        landmarks_3d_count = len(landmarks_map.COORDS_3D)

        points_2d = np.zeros((landmarks_3d_count, 2), dtype='double')
        points_3d = np.zeros((landmarks_3d_count, 3), dtype='double')
        for l_idx, name in enumerate(landmarks_map.COORDS_3D.keys()):
            points_2d[l_idx, :] = landmarks_map.get(name)
            points_3d[l_idx, :] = landmarks_map.get_3d(name)

        start_t = time.time()
        (success, self.rot, self.trans) = cv2.solvePnP(
            points_3d, points_2d, self.camera_matrix, self.dist_coeffs
        )
        if not success:
            raise ValueError('Could not estimate projection')

    def project_to_2d(self, x, y, z):
        point_2d, _ = cv2.projectPoints(
            np.array([(x, y, z)], dtype='float64'),
            self.rot, self.trans, self.camera_matrix, self.dist_coeffs
        )

        return point_2d[0][0].astype(int)

    def get_cam_pos(self):
        # convert from opencv type to rotation 3x3 matrix
        R = cv2.Rodrigues(self.rot)[0]
        # apply -R^t * T to get 3x1 camera position vector
        pos = (-R).T.dot(self.trans)
        return pos

    def get_cam_target_pos(self):
        # convert from opencv type to rotation 3x3 matrix, transpose it
        R = cv2.Rodrigues(self.rot)[0].T
        point_at = R.dot([0, 0, 1])
        return point_at
        scipy_rot = Rotation.from_matrix(R)
        # panda3d requires euler angles for rotation
        return scipy_rot.as_euler('zxy', degrees=True)

    def get_cam_roll(self):
        # convert from opencv type to rotation 3x3 matrix, transpose it
        R = cv2.Rodrigues(self.rot)[0].T
        scipy_rot = Rotation.from_matrix(R)
        # panda3d requires euler angles for rotation
        roll = scipy_rot.as_euler('zxy', degrees=True)[0]
        return roll
