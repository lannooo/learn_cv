import math
import random
from drawobj import *

_fps = 30.0
_wait_time = int(1000 / _fps)
# BE CAREFUL!! The window size of videowriter should be the same as the generated image/frame
_height = 600
_width = 800
_window_size = (_height, _width)  # height, width
_frame_size = (_width, _height)  # width, height
_background_rgb = (*_window_size, 3)
WIN_NAME = 'Homework-1'


class Stage:
    def __init__(self, window_size, canvas_size, loc=None, back_color=COLOR_BLACK):
        self.back_color = back_color
        self.window_size = window_size
        self.canvas_size = canvas_size
        if loc is None:
            loc = (0, 0)
        self.loc = loc
        self.objects = []

    def remove_object(self, index):
        return self.objects.pop(index)

    def clear_objects(self):
        self.objects.clear()

    def add_object(self, obj):
        self.objects.append(obj)

    def add_objects(self, objs):
        self.objects.extend(objs)

    def loc_move(self, dx, dy):
        self.loc = (self.loc[0] + dy, self.loc[1] + dx)

    def draw(self):
        layer = background(self.canvas_size, self.back_color)
        for o in self.objects:
            layer = o.draw(layer)
        y0 = self.loc[0]
        x0 = self.loc[1]
        # cut only the window size
        return layer[y0:y0 + self.window_size[0], x0:x0 + self.window_size[1]]

    def update_frame(self, frame_cnt):
        pass


class Switch:
    def __init__(self):
        self.f1 = None
        self.f2 = None

    def set_before_frame(self, f):
        self.f1 = f

    def set_after_frame(self, f):
        self.f2 = f

    def draw(self):
        pass

    def update_frame(self, frame_cnt):
        pass

class Switch_Blackhole(Switch):
    def __init__(self):
        super(Switch_Blackhole, self).__init__()
        self.action_frames = 100
        self.angle = 0
        self.scale = 1.0

    def draw(self):
        return image_rotate(self.f1, (_width//2, _height//2), self.angle, self.scale)

    def update_frame(self, frame_cnt):
        if frame_cnt < self.action_frames:
            self.angle += 7
            self.scale = self.scale * math.pow(0.999, frame_cnt)
            return True
        else:
            return False


class Switch_Fade(Switch):
    def __init__(self):
        super(Switch_Fade, self).__init__()
        self.w = 1.0
        self.action_frames = 30
        self.step = self.w / self.action_frames

    def draw(self):
        assert self.f1 is not None and self.f2 is not None
        # f1 fade out and f2 fade in
        return image_add(self.f1, self.f2, weight=(self.w, 1-self.w))

    def update_frame(self, frame_cnt):
        if frame_cnt < self.action_frames:
            self.w -= self.step
            return True
        else:
            return False

def random_color(main_color):
    s = sum(main_color)
    c1 = random.randint(s//6, 2*s//3)
    s -= c1
    c2 = random.randint(s//6, 2*s//3)
    c3 = s-c2
    return (c1, c2, c3)

class Stage4(Stage):
    def __init__(self):
        super(Stage4, self).__init__(_window_size,
                                     (_height, _width),
                                     back_color=COLOR_BLACK)
        self.end_text = TextObject("谢谢！", font(0), COLOR_WHITE, (_height//2, _width//2))
        self.end_list = TextObject("编剧：王二狗\n导演：老王\n主演1：赛博小蛇\n主演2：赛博小球\n执行：Python\n灯光：胡乱一搞\n",
                                   font(26), COLOR_WHITE, (_height, _width//2))
    def update_frame(self, frame_cnt):
        if frame_cnt == 0:
            self.add_object(self.end_text)
        elif frame_cnt < 50:
            self.end_text.change_font(frame_cnt)
        elif frame_cnt == 50:
            self.add_object(self.end_list)
        elif frame_cnt < 350:
            self.end_text.move_delta(0, -3)
            self.end_list.move_delta(0, -3)
        else:
            return False
        return True

class Stage3(Stage):
    def __init__(self):
        super(Stage3, self).__init__(_window_size,
                                     (_height, _width),
                                     back_color=COLOR_WHITE)
        self.center = ( _width // 2, _height//2)
        self.heart = CircleObject(self.center, radius=(0, 0), color=COLOR_1, thickness=-1)
        self.add_object(self.heart)
        self.radius_seq = self.r_seq()

    def r_seq(self):
        t1 = np.linspace(-0.5*np.pi, 0.5*np.pi, num=30)
        seq1 = np.sin(t1)+1
        t2 = np.linspace(0, 2*np.pi, num=30)
        seq2 = (np.cos(t2)+1)*0.5+1
        return np.concatenate([seq1, seq2, seq2, seq2, seq2, seq2, seq2, seq2, seq2, seq2, seq2])

    def update_frame(self, frame_cnt):
        if frame_cnt>=0 and frame_cnt<300:
            if len(self.objects) == 0:
                self.add_object(self.heart)
            if frame_cnt <= 200:
                r = int(self.radius_seq[frame_cnt] * 10)
                self.heart.change_radius((r, r))
            if frame_cnt > 100:
                if frame_cnt <= 200:
                    r2 = random.randint(30, 45)
                    o = CircleObject(self.center, radius=(r2, r2), color=random_color(COLOR_2), thickness=random.randint(1, 4))
                    self.add_object(o)
                for i in range(1, len(self.objects)):
                    self.objects[i].add_radius(random.randint(0, 10))
            if frame_cnt == 200:
                self.add_object(self.remove_object(0))
            if frame_cnt > 200:
                self.heart.add_radius(7)
            return True
        else:
            return False

class Stage2(Stage):
    def __init__(self):
        super(Stage2, self).__init__(_window_size,
                                     (1200, 1600),
                                     loc=(300, 400),
                                     back_color=COLOR_1)
        self.bug_nodes = 0
        self.max_nodes = 7
        self.gate = RectangleObject((750, 560), (50, 50), COLOR_GREEN, thickness=2)
        self.bean = CircleObject((760 + (self.max_nodes) * 35 + 10, 570-300), (25, 25), COLOR_YELLOW, -1)
        self.add_object(self.gate)
        self.add_object(self.bean)

    def new_bug_node(self, pos):
        return RectangleObject(
            pos,
            (30, 30),
            COLOR_4,
            thickness=-1
        )

    def update_frame(self, frame_cnt):
        if frame_cnt > self.bug_nodes*10 and frame_cnt <= self.max_nodes*10:
            self.bug_nodes += 1
            node = self.new_bug_node((760+(frame_cnt//10)*35, 570))
            self.add_object(node)
        elif frame_cnt > self.bug_nodes*10 and frame_cnt <= self.max_nodes*10 + 30:
            self.bug_nodes += 1
            self.remove_object(2)
            node = self.new_bug_node((760 + (self.max_nodes) * 35, 570 - (frame_cnt//10 - self.max_nodes) * 35))
            self.add_object(node)
        elif frame_cnt > self.bug_nodes*10 and frame_cnt <= self.max_nodes*10 + 90:
            self.bug_nodes += 1
            self.remove_object(2)
            node = self.new_bug_node((760 + (self.max_nodes) * 35, 570 - (frame_cnt//10 - self.max_nodes) * 35))
            self.add_object(node)
            self.loc_move(0, -35)
        elif frame_cnt > 170 and frame_cnt <= 170+self.max_nodes:
            self.remove_object(2)
        elif frame_cnt > 170+self.max_nodes and frame_cnt <= 240:
            # self.back_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.bean.change_radius((25+int(math.pow(1.2, frame_cnt-170)), 25+int(math.pow(1.2, frame_cnt-170))))

        elif frame_cnt > 220:
            return False

        return True


class Stage1(Stage):
    def __init__(self):
        super(Stage1, self).__init__(_window_size, (_height * 2, _width * 2), back_color=COLOR_WHITE)
        logo = cv.imread("img/logo.png")
        resized_person = cv.resize(cv.imread("img/personal.jpg"), None, fx=0.5, fy=0.5)
        self.logo = ImageObject(logo, (200, 580))
        self.title = TextObject("OpenCV", font(128), COLOR_ZJU_BLUE, (200, 20))
        self.sepline = LineObject((20, 375), (520, 375), COLOR_ZJU_BLUE, thickness=2)
        self.desc = TextObject("姓名：王二狗\n学号：00000000\n学院：计算机科学与技术学院",
                               font(24), COLOR_ZJU_BLUE, (600, 400))
        self.person = ImageObject(resized_person, (600, 200), thresh=254)
        self.add_objects([self.logo, self.title, self.sepline, self.desc, self.person])

    def update_frame(self, frame_cnt):
        if frame_cnt <= 40 and frame_cnt >= 10:
            self.loc_move(0, 5)
        if frame_cnt >= 50:
            return False
        return True


if __name__ == '__main__':
    # Define the codec and create VideoWriter object
    print(f"Run in optimized: {cv.useOptimized()}")
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    video_writer = cv.VideoWriter('opencv_homework1.mp4', fourcc, _fps, _frame_size)
    cv.namedWindow(WIN_NAME)
    frame_cnt = 0
    # the whole video sequence, consists some Stages and Switches
    stages = [Stage1(), Switch_Fade(), Stage2(), Switch_Fade(), Stage3(), Switch_Blackhole(), Stage4()]
    stage_cnt = 0
    while True:
        # generate a frame
        frame = stages[stage_cnt].draw()
        # update next frame parameters
        do_more = stages[stage_cnt].update_frame(frame_cnt)
        if not do_more:
            frame_cnt = 0
            stage_cnt += 1
            # reach end, stop
            if stage_cnt == len(stages): break
            # Switch type need to give the last frame and the next frame to make some animation
            if isinstance(stages[stage_cnt], Switch):
                stages[stage_cnt].set_before_frame(frame)
                stages[stage_cnt].set_after_frame(stages[stage_cnt + 1].draw())
        else:
            frame_cnt += 1

        # show frame and write frame to video_writer
        video_writer.write(frame)
        cv.imshow(WIN_NAME, frame)
        # keyboard control logic
        key = cv.waitKey(_wait_time)
        if key == ord(' '):
            # press space to pause, and press it again to resume
            while cv.waitKey(-1) != ord(' '):
                pass
        elif key & 0xFF == 27 or key == ord('q'):
            # ESC pressed or 'q' pressed exit
            break

    # Release everything if job is finished
    video_writer.release()
    cv.destroyAllWindows()
