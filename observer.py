import os
from multiprocessing import Process, Queue
import keyboard
import pyautogui
import time
import tkinter as tk
from mss import mss
import numpy as np
import cv2
import math
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
WINDOW_WIDTH = 2880
WINDOW_HEIGHT = 1800
#GAME_LOCATION = {"left": 1125, "top": 507, "width": 633, "height": 774}
GAME_LOCATION = {"left": 1137, "top": 954, "width": 657, "height": 216}
MIDDLE_TREE_X = 1422
LEFT_SIDE =  {"left": 1140, "top": 892, "width": 199, "height": 150}
RIGHT_SIDE = {"left": 1541, "top": 892, "width": 199, "height": 150}


IS_LEFT_BRANCH = {"left":1130,"top":1135,"width":186,"height":65}
IS_RIGHT_BRANCH = {"left":1547,"top":1135,"width":186,"height":65}

GAME_OVER_LOCATION = {"left":1334,"top":503,"width":93,"height":169}
class Position:
    def __init__(self,x,y,w,h,side):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.side = side
    def __str__(self):
        return f"x={self.x};y={self.y};side={self.side}"


class Laser:
    def __init__(self,start_x,start_y,end_x,end_y=10):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

# 0 - left
# 1 - right
# 2  - no branch
class Observer:
    def __init__(self,q,data_q):
        self.q = q
        self.data_queue = data_q
        self.face_color = (50, 194, 247) # GUY WITH YELLOW HELMET
        # by pattern GUY WITH YELLO SUIT
        self.frame_color =  (34, 113, 124)#140203
        self.game_over_color = (210,241,252)
        self.branch_color = [
            (196, 200, 91),  # #5BC8C4
            (168, 170, 73),  # #49AAA8
            (117, 113, 35),  # #237175
            (90, 135, 36),  # #24875A
            (121, 173, 50),  # #32AD79
            (123, 176, 51),  # #33B07B
            (16, 89, 142),  # #8E5910
            (23, 128, 204),  # #CC8017
            (146, 132, 79),  # #4F8492
            (34, 113, 124),  # #7C7122
            (63, 150, 163),  # #A3963F
            (113, 170, 255),  # #FFAA71
            (71, 139, 241),  # #F18B47
            (41, 82, 204),  # #CC5229
            (141, 138, 114),  # #728A8D
            (193, 194, 168),  # #A8C2C1
        ]
        self.preload_template = []
        for template in os.listdir("other"):
            path = os.path.join("other",template)
            img =cv2.imread(path)
            if img is not None:
                self.preload_template.append(img)



    def find_color_match(self,frame,color,transformation,tolerance):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        lower = np.array([max(0,c-tolerance) for c in color])
        upper = np.array([min(255,c+tolerance) for c in color])

        mask = cv2.inRange(frame, lower, upper)
        points = np.column_stack(np.where(mask > 0))
        if points.size > 0:
            y, x = points[0]
            x,y = transformation(x,y)
            w, h = 10, 10
            if x < MIDDLE_TREE_X:
                side = 0
            else:
                side = 1
            return Position(x,y,w,h,side)
        return None

    def find_pattern(self,frame,transformation):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        for template in self.preload_template:
                res = cv2.matchTemplate(frame_bgr,template,cv2.TM_CCOEFF_NORMED)
                _,max_val,_,max_loc = cv2.minMaxLoc(res)
                x, y = max_loc
                x,y = transformation(x,y)
                if max_val>0.6:
                    #print(f"TOP_LEFT: X={x}  Y={y}")
                    if x < MIDDLE_TREE_X:
                        return Position(x,y,10,10,0)
                    if x >= MIDDLE_TREE_X:
                        return Position(x,y,10,10,1)
        return None

    def draw_laser(self,head_x,head_y,frame,side:int):
        for color in self.branch_color:
            if side == 0:
                p = self.find_color_match(frame,color,transform_left_side,tolerance=0)
                if p is not None:
                   self.q.put(Laser(head_x,head_y,head_x,p.y))
                   return head_y - p.y
            if side == 1:
                p = self.find_color_match(frame,color,transform_right_side,tolerance=0)
                if p is not None:
                    self.q.put(Laser(head_x, head_y, head_x, p.y))
                    return head_y - p.y
        return float('inf')


# 0 -> left
# 1 -> right
# 2 -> no branch on this lvl
    def is_branch_on_lvl(self,frame,side): # side is the opposite of character side
        for color in self.branch_color:
            if side == 0:
                p = self.find_color_match(frame,color,transform_is_left_side,tolerance=0)
                if p is not None:
                    return side
            if side == 1:
                p = self.find_color_match(frame, color, transform_is_right_side, tolerance=0)
                if p is not None:
                    return side

        return 2

    def is_game_over(self,frame):
        p = self.find_color_match(frame,self.game_over_color,transform_game_location,tolerance=0) # we dont have to get true x y
        return not p is None
    def process_screen(self):
        with mss() as sccp:
            character_pos = None
            branch_pos = None
            branch_distance = None
            while True:
                frame = np.array(sccp.grab(GAME_LOCATION))
                m = self.find_color_match(frame,self.face_color,transform_game_location,tolerance=0)
                #m = self.find_pattern(frame,transform_game_location)
                if m is not None:
                    character_pos = m.side
                    self.q.put(m)
                    if m.side == 0 : #left
                        frame = np.array(sccp.grab(LEFT_SIDE))
                        branch_distance =  self.draw_laser(m.x,m.y,frame,0)
                        frame = np.array(sccp.grab(IS_RIGHT_BRANCH))
                        branch_pos = self.is_branch_on_lvl(frame,1)
                    if m.side == 1: #right
                        frame = np.array(sccp.grab(RIGHT_SIDE))
                        branch_distance =self.draw_laser(m.x,m.y,frame,1)
                        frame = np.array(sccp.grab(IS_LEFT_BRANCH))
                        branch_pos = self.is_branch_on_lvl(frame,0)
                self.data_queue.put([character_pos,branch_pos,branch_distance])
                frame = np.array(sccp.grab(GAME_OVER_LOCATION))
                print(self.is_game_over(frame))
                time.sleep(0.1)




def center_window():
    return f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

def transform_game_location(x, y):
    return GAME_LOCATION['left']+x,GAME_LOCATION['top']+y

def transform_left_side(x, y):
    return LEFT_SIDE['left']+x,LEFT_SIDE['top']+y

def transform_right_side(x, y):
    return RIGHT_SIDE['left']+x,RIGHT_SIDE['top']+y

def transform_is_left_side(x, y):
    return IS_LEFT_BRANCH['left']+x,IS_LEFT_BRANCH['top']+y

def transform_is_right_side(x, y):
    return IS_RIGHT_BRANCH['left']+x,IS_RIGHT_BRANCH['top']+y

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
        self.line_id = None
        self.find_match_process = None
        self.queue = Queue()
        self.data_queue = Queue()
        self.observer = Observer(self.queue,self.data_queue)

        self.start_background_process()
        self.root.after(1, self.check_queue)
        self.root.after(1,self.check_data_queue)


    def start_background_process(self):
        self.find_match_process = Process(target=self.observer.process_screen, args=())
        self.find_match_process.start()

    def check_queue(self):# add match to types
        while not self.queue.empty():
            pos = self.queue.get()

            match pos:
                case Position():
                    if self.rect_id is None:
                        self.rect_id = self.canvas.create_rectangle(pos.x, pos.y, pos.x + pos.w, pos.y + pos.h, fill='red')
                    else:
                        self.canvas.coords(self.rect_id, pos.x, pos.y, pos.x + pos.w, pos.y + pos.h)
                case Laser():
                    if self.line_id is None:
                        self.line_id = self.canvas.create_line(pos.start_x, pos.start_y, pos.end_x , pos.end_y, fill='red',width=2)
                    else:
                        self.canvas.coords(self.line_id, pos.start_x, pos.start_y, pos.end_x , pos.end_y)



        self.root.after(1, self.check_queue)


    def check_data_queue(self):
        while not self.data_queue.empty():
            pos = self.data_queue.get()
            print(pos)
        self.root.after(1, self.check_data_queue)

    def run(self):
        self.root.mainloop()
        self.find_match_process.terminate()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    app.run()


