from draw import *


class Object:
    def draw(self, layer):
        pass

    def change_pos(self, pos):
        pass

    def move_delta(self, dx, dy):
        pass


def tuple_add(t, d1, d2):
    return t[0] + d1, t[1] + d2


class TextObject(Object):
    def __init__(self, text, font, color, position):
        super(TextObject, self).__init__()
        self.text = text
        self.font = font
        self.color = color
        self.left = position[1]
        self.top = position[0]

    def draw(self, layer):
        return draw_text_truetype(layer, self.text, self.left, self.top, self.font, self.color)

    def change_pos(self, pos):
        self.left = pos[1]
        self.top = pos[0]

    def move_delta(self, dx, dy):
        self.left += dx
        self.top += dy

    def change_font(self, font_size):
        self.font = font(font_size)


class ImageObject(Object):
    def __init__(self, image, position, thresh=200, inverse=True):
        super(ImageObject, self).__init__()
        # for white background color image, to replace the back-color to black first
        self.image = image_combine_roi(background(image.shape[:3], COLOR_BLACK), image, thresh=thresh, inverse=inverse)
        self.position = position

    def draw(self, layer):
        return image_combine_roi(layer, self.image, self.position, thresh=0)

    def change_pos(self, pos):
        self.position = pos

    def move_delta(self, dx, dy):
        self.position = tuple_add(self.position, dy, dx)


class CircleObject(Object):
    def __init__(self, position, radius: tuple, color, thickness, angle=0):
        self.center = position
        self.radius = radius
        self.angle = angle
        self.color = color
        self.thickness = thickness

    def draw(self, layer):
        if self.radius[0] == self.radius[1]:
            return draw_circle(layer, self.center, self.radius[0], self.color, self.thickness)
        else:
            return draw_ellipse(layer, self.center, self.radius, self.angle, 0, 360, self.color, self.thickness)

    def change_pos(self, pos):
        self.center = pos

    def move_delta(self, dx, dy):
        self.center = tuple_add(self.center, dx, dy)

    def change_radius(self, radius):
        self.radius = radius
    def add_radius(self, delta):
        self.radius = (self.radius[0]+delta, self.radius[1]+delta)
    def change_angle(self, angle):
        self.angle = angle


class PolyObject(Object):
    def __init__(self, pts, color, close=True):
        self.pts = []
        self.pts.extend(pts)
        self.color = color
        self.close = close

    def draw(self, layer):
        return draw_polylines(layer, self.pts, self.close, self.color)

    def move_delta(self, dx, dy):
        for i in range(len(self.pts)):
            self.pts[i] = tuple_add(self.pts[i], dx, dy)


class LineObject(Object):
    def __init__(self, start, end, color, thickness):
        self.thickness = thickness
        self.end = end
        self.start = start
        self.color = color

    def draw(self, layer):
        return draw_line(layer, self.start, self.end, self.color, self.thickness)

    def change_pos(self, pos):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.start = (pos[1], pos[0])
        self.end = tuple_add(self.start, dx, dy)

    def move_delta(self, dx, dy):
        self.start = tuple_add(self.start, dx, dy)
        self.end = tuple_add(self.end, dx, dy)


class RectangleObject(Object):
    """rectangle from start (x, y) and size (dx, dy)"""
    def __init__(self, start, size, color, thickness):
        self.thickness = thickness
        self.start = start
        self.end = (start[0]+size[0], start[1]+size[1])
        self.color = color

    def draw(self, layer):
        return draw_rectangle(layer, self.start, self.end, self.color, self.thickness)

    def change_pos(self, pos):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.start = (pos[1], pos[0])
        self.end = tuple_add(self.start, dx, dy)

    def move_delta(self, dx, dy):
        self.start = tuple_add(self.start, dx, dy)
        self.end = tuple_add(self.end, dx, dy)
