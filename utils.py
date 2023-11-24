import os
import json

import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib.pyplot as plt


# 540, 960

def _get_area2zones(areas, zones):
    areas_polygons = [Polygon(x) for x in areas]
    area2zones = {k: [] for k in range(len(areas_polygons))}
    for i in range(len(zones)):
        center = Point(*zones[i].mean(axis=0))

        did_found = False
        for j in range(len(areas_polygons)):
            if areas_polygons[j].contains(center):
                area2zones[j].append(zones[i])
                did_found = True
                break
        if not did_found:
            raise ValueError("Zone was not located inside area")

    for i in range(len(area2zones)):
        if len(area2zones[i]) != 2:
            raise ValueError("Area does not have 2 zones")
    return area2zones


def _get_cross_orientation(zone):
    orientation = None
    max_len = -1
    for i in range(len(zone)):
        direction_vector = zone[(i + 1) % 4] - zone[i]
        dist = np.linalg.norm(direction_vector, ord=2)
        if dist > max_len:
            max_len = dist
            orientation = direction_vector / dist
    return orientation


def get_oriented_annotations(annotation):
    areas = np.array(annotation['areas'], dtype=np.float32)
    ini_zones = np.array(annotation['zones'], dtype=np.float32)
    area2zones = _get_area2zones(areas, ini_zones)
    out_areas = []
    for i, area in enumerate(areas):
        zones = area2zones[i]
        orientation = _get_cross_orientation(zones[0])
        start_index = 0
        for j in range(len(area)):
            c_vec = area[(j + 1) % 4] - area[j]
            c_vec = c_vec / np.linalg.norm(c_vec, 2)
            if abs(np.dot(orientation, c_vec)) > 0.8:
                start_index = j
                break
        new_area = np.roll(area, shift=-start_index, axis=0)
        out_areas.append(new_area)

    return np.array(out_areas)


def distance_vector(p: np.array, a: np.array, b: np.array):
    ap = p - a
    ab = b - a
    projection = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return projection - p


def lies_between(p: np.array, seg1: np.ndarray, seg2: np.ndarray):
    assert seg1.shape[0] == 2
    assert seg2.shape[0] == 2
    dist1 = distance_vector(p, seg1[0], seg1[1])
    dist2 = distance_vector(p, seg2[0], seg2[1])
    if np.dot(dist1, dist2) < 0:
        return True
    else:
        return False


def mid_projection(center: np.array, mid_par: np.ndarray):
    ap = center - mid_par[0]
    ab = mid_par[1] - mid_par[0]
    proj = np.dot(ap, ab) / np.dot(ab, ab)

    # proj = np.dot(ap, ab) / np.dot(ab, ab) * ab
    # proj = np.linalg.norm(proj) / np.linalg.norm(ab)

    return proj


if __name__ == "__main__":
    a = np.array([0, 0])
    b = np.array([5, 5])
    p = np.array([5, 0])
    print(distance_vector(p, a, b))

    seg1 = np.array([[0, 0], [5, 5]])
    seg2 = np.array([[-5, 0], [0, 7]])
    p = np.array([999, 1000])
    print(lies_between(p, seg1, seg2))