import cv2
import time

from face import Face
from lenses import GlassesLens
from processor import Processor


def main_one():
    img = cv2.imread('/Users/anton.kasyanov/Desktop/pic.jpg')

    face = Face(img)

    glasses_lense = GlassesLens()
    img = glasses_lense.overlay(face)

    cv2.imshow('preview', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


ASYNC = False

def main_cam():
    cam = cv2.VideoCapture(2)
    processor = Processor()
    glasses_lense = GlassesLens()

    frames_t = []  # timestamp of each frame

    while True:
        ret, img = cam.read()

        if ASYNC:
            processor.feed_frame(img)
            img = processor.get_frame(blocking=False)
        else:
            face = Face(img)
            img = glasses_lense.overlay(face)

        # FPS
        cur_t = time.time()
        frames_t.append(cur_t)
        # remove frames that are more then 1 sec old
        frames_t = [t for t in frames_t if cur_t - t < 1]
        cv2.putText(
            img, f'FPS: {len(frames_t)}', (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

        cv2.imshow('preview', img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cam.release()
    processor.stop()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # main_one()
    main_cam()
