import queue
from multiprocessing import Process, Queue

from lenses import GlassesLens
from face import Face


STOP_COMMAND = 'stop'
PROCESS_COMMAND = 'process'


def _proc_func(task_q, res_q):

    while True:
        # wait until got a task
        task = task_q.get(block=True)
        if task[0] == STOP_COMMAND:
            # need to stop
            break

        if task[0] != PROCESS_COMMAND:
            raise ValueError(f'Bad command {task[0]}')

        img = task[1]
        face = Face(img)

        res_q.put(face)


class Processor:

    def __init__(self):
        self._last_frame = None  # no face or anything
        self._last_face = None  # last processed face, potentially lags
        self._process_busy = False

        self.task_q = Queue()  # parent to process
        self.res_q = Queue()
        self.p = Process(target=_proc_func, args=(self.task_q, self.res_q,))
        self.p.start()

        self.glasses_lens = GlassesLens()

    def feed_frame(self, img):
        self._last_frame = img.copy()

        if not self._process_busy:
            self.task_q.put((PROCESS_COMMAND, self._last_frame))
            self._process_busy = True

    def get_frame(self, blocking=False):
        try:
            # get processed face from the process
            self._last_face = self.res_q.get(blocking)
            self._process_busy = False
        except queue.Empty:
            # if not ready, just use the latest
            pass

        res = self._last_frame
        if self._last_face:
            res = self.glasses_lens.overlay(self._last_face, self._last_frame)
        return res

    def stop(self):
        self.task_q.put((STOP_COMMAND,))
        self.p.join(timeout=1)
