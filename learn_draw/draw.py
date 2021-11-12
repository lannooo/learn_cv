import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# B, G, R
COLOR_1 = (38, 1, 6)
COLOR_2 = (155, 242, 5)
COLOR_3 = (175, 242, 5)
COLOR_4 = (199, 242, 5)
COLOR_5 = (101, 140, 3)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY_1 = (100, 100, 100)
COLOR_ZJU_BLUE = (173, 63, 3)
COLOR_YELLOW = (0, 255, 255)

_font_map = {}
_font_default = ImageFont.truetype("font/微软雅黑.ttc", 26, encoding="utf-8")
_font_map[26] = _font_default


def font(size):
    if size not in _font_map:
        _font_map[size] = ImageFont.truetype("font/微软雅黑.ttc", size, encoding="utf-8")
    return _font_map[size]


def draw_text_truetype(img, text, left, top, font_type=_font_default, color=COLOR_WHITE):
    """draw chinese text on specific position"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    rgb_color = (color[2], color[1], color[0])
    draw.text((left, top), text, rgb_color, font=font_type)
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def draw_line(img, start, end, color=COLOR_WHITE, thickness=1):
    # direct modification
    cv.line(img, start, end, color, thickness)
    return img


def draw_rectangle(img, start, end, color=COLOR_WHITE, thickness=1):
    # direct modification
    cv.rectangle(img, start, end, color, thickness)
    return img


def draw_circle(img, center, radius, color=COLOR_WHITE, thickness=1):
    # direct modification
    cv.circle(img, center, radius, color, thickness)
    return img


def draw_ellipse(img, center, axis, angle=0, angle_s=0, angle_e=360, color=COLOR_WHITE, thickness=1):
    # direct modification
    cv.ellipse(img, center, axis, angle, angle_s, angle_e, color, thickness)
    return img


def draw_polylines(img, pts, close=True, color=COLOR_WHITE):
    # direct modification
    points = np.array(pts, np.int32)
    points = points.reshape((-1, 1, 2))
    cv.polylines(img, [points], close, color)
    return img


def image_replace(back_img, img, pos):
    h = img.shape[0]
    w = img.shape[1]
    # copy to avoid direct modification
    back_img = np.copy(back_img)
    back_img[pos[0]:pos[0]+h, pos[1]:pos[1]+w] = img
    return back_img


def image_add(img1, img2, pos=(0, 0), weight=None):
    rows, cols, channels = img2.shape
    temp = np.zeros_like(img1, dtype=np.uint8)
    temp[pos[0]:pos[0]+rows, pos[1]:pos[1]+cols] = img2
    img2 = temp
    if weight is not None:
        w1, w2 = weight
        return cv.addWeighted(img1, w1, img2, w2, 0)
    else:
        return cv.add(img1, img2)


def image_combine_roi(img1, img2, pos=(0, 0), thresh=10, inverse=False):
    """draw a small image inside bigger background naturally"""
    rows, cols, channels = img2.shape
    # copy to avoid direct modification
    img1 = np.copy(img1)
    roi = img1[pos[0]:pos[0] + rows, pos[1]:pos[1] + cols]
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    if inverse:
        type = cv.THRESH_BINARY_INV
    else:
        type = cv.THRESH_BINARY
    ret, mask = cv.threshold(img2_gray, thresh, 255, type)
    mask_inv = cv.bitwise_not(mask)
    img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv.bitwise_and(img2, img2, mask=mask)
    dst = cv.add(img_bg, img_fg)
    img1[pos[0]:pos[0] + rows, pos[1]:pos[1] + cols] = dst
    return img1


def background(shape, color=COLOR_WHITE):
    if len(shape) == 2:
        shape = (shape[0], shape[1], 3)
    img = np.empty(shape, np.uint8)
    img[:, :] = color
    return img


def image_move(img, x, y):
    transform = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(img, transform, (img.shape[1], img.shape[0]))


def image_rotate(img, center, angle, scale):
    transform = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(img, transform, (img.shape[1], img.shape[0]))


def image_affine(img, p1, p2):
    transform = cv.getAffineTransform(p1, p2)
    return cv.warpAffine(img, transform, (img.shape[1], img.shape[0]))
