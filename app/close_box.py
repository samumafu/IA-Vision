"""Refine an existing detection with the outline of a nearby object."""
import cv2
import numpy as np


def refine_close_box(rgb, box):
    height, width = rgb.shape[:2]
    scale = 384 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if min(size) < 18:
        return box
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    edge_margin = max(2, round(min(sh, sw) * .02))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, 5, 30, 25)
    kernel = np.ones((5, 5), dtype=np.uint8)
    left, top, right, bottom = box
    area = (right - left) * (bottom - top)
    best = (0., box)
    for threshold in (22, 35, 50):
        edges = cv2.Canny(smooth, threshold, 3 * threshold)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            pixels = cw * ch
            if (min(cw, ch) < 18 or not 750 <= pixels <= .78 * sh * sw or
                    min(x, y, sw - x - cw, sh - y - ch) < edge_margin):
                continue
            fill = cv2.contourArea(contour) / pixels
            if fill < .35:
                continue
            candidate = (x * width / sw, y * height / sh, (x + cw) * width / sw, (y + ch) * height / sh)
            ax, ay, bx, by = candidate
            size_ratio = (bx - ax) * (by - ay) / area
            if not 1.15 <= size_ratio <= 4.:
                continue
            overlap_width = max(0., min(bx, right) - max(ax, left))
            overlap_height = max(0., min(by, bottom) - max(ay, top))
            coverage = overlap_width * overlap_height / area
            if coverage >= .75:
                best = max(best, (fill * min(2., size_ratio) * coverage, candidate))
    return best[1]
