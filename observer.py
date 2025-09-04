from multiprocessing import Process, Queue
import keyboard
import pyautogui
import time
import tkinter as tk
from mss import mss
import numpy as np
import cv2

WINDOW_WIDTH = 2880
WINDOW_HEIGHT = 1800
game_location = {"left": 1125, "top": 507, "width": 633, "height": 774}

class Observer:
    def __init__(self,q):
        self.q = q
        self.target_color = (83, 171, 165)  # #53ABA5 w BGR



    def find_match(self):
        with mss() as sccp:
            while True:
                frame = np.array(sccp.grab(game_location))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                lower = np.array(self.target_color)
                upper = np.array(self.target_color)
                mask = cv2.inRange(frame, lower, upper)
                print(frame)
                points = np.column_stack(np.where(mask > 0))
                if points.size > 0:
                    y, x = points[0]
                    h, w = 10, 10
                    self.q.put((x, y, w, h))

                time.sleep(0.001)



def center_window():
    return f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

def transform(x, y):
    return game_location['left']+x,game_location['top']+y


class App:
    def __init__(self, root):
        self.root = root
        root.tk.call('tk', 'scaling', 1.0)
        root.attributes("-fullscreen", True)
        root.geometry(center_window())
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", "black")
        root.configure(bg="black")
        self.canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                                bg="black", highlightthickness=0)
        self.canvas.pack()
        root.bind('q', lambda e: root.quit())
        root.focus_set()
        self.rect_id = None
        self.process = None
        self.queue = Queue()
        self.observer = Observer(self.queue)

        self.start_background_process()
        self.root.after(1, self.check_queue)

    def start_background_process(self):
        self.process = Process(target=self.observer.find_match, args=())
        self.process.start()

    def check_queue(self):
        while not self.queue.empty():
            x, y,_,_ = self.queue.get()
            x,y = transform(x,y)
            if self.rect_id is None:
                self.rect_id = self.canvas.create_rectangle(x,y,x+10,y+10,fill='red')
            else:
                self.canvas.coords(self.rect_id,x,y,x+10,y+10)

        self.root.after(1, self.check_queue)

    def run(self):
        self.root.mainloop()
        self.process.terminate()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    app.run()


