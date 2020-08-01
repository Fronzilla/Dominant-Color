# -*- coding: utf-8 -*-
""""
Finding dominant color for an image
"""

import numpy as np
import cv2
import webcolors

from functools import lru_cache
from typing import List, Tuple, NoReturn, Union
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

from helpers import *

__author__ = 'Aleksey Nikitin'
__version__ = '0.0.1'

warnings.warn = ignore_warn


class AverageStrategy:
    """ An average strategy for extracting an average color of picture """

    @accepts()
    def __init__(self, image: str):
        self._rgb: Tuple[int] = tuple()

        img = cv2.imread(image)
        height, width, _ = np.shape(img)

        # calculate the average color of each row of our image
        avg_color_per_row = np.average(img, axis=0)

        # calculate the averages of our rows
        avg_colors = np.average(avg_color_per_row, axis=0)

        self._rgb = np.array(avg_colors, dtype=np.uint8)

    @property
    def rgb(self):
        return self._rgb

    @staticmethod
    @lru_cache(128)
    def _get_colour() -> Tuple[List[str], List[webcolors.IntegerRGB]]:
        """
        Storing color names combination
        """

        # Populating color names into spatial name database
        hex_names = webcolors.CSS3_HEX_TO_NAMES
        names = []
        positions = []

        for hex_, name in hex_names.items():
            names.append(name)
            positions.append(webcolors.hex_to_rgb(hex_))

        return names, positions

    def get_colour(self):
        """ Finding color using nearest neighborhood algorithm """
        # lets populate some names into spatial name database
        names, positions = self._get_colour()

        kd_tree = KDTree(positions)

        # query nearest point
        dist, index = kd_tree.query(self.rgb)

        log.info(f'The color {self.rgb} is closest to {names[index]}.')

        return names[index]


class KMeansClusteringStrategy:
    """ KMeans Clustering realisation for extracting dominant color """

    @accepts()
    def __init__(self, image: str, all_colors: bool = False, num_clusters=5):

        self._rgb: Union[Tuple[int], List[Tuple[int]]] = []
        self._hsv: Union[Tuple[int], List[Tuple[int]]] = []

        img = cv2.imread(image)
        height, width, _ = np.shape(img)

        # reshape the image to a list of RGB pixels
        image = img.reshape((height * width, 3))

        self.clusters = KMeans(n_clusters=num_clusters)
        self.clusters.fit(image)

        self.make_histogram(all_colors)

    @property
    def rgb(self) -> Tuple[int]:
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        self._rgb = value

    @property
    def hsv(self) -> Tuple[int]:
        return self._hsv

    @hsv.setter
    def hsv(self, value):
        self._hsv = value

    @staticmethod
    @lru_cache(128)
    def _get_colour() -> Tuple[List[str], List[webcolors.IntegerRGB]]:
        """ Storing color names combination """

        # Populating color names into spatial name database
        hex_names = webcolors.CSS3_HEX_TO_NAMES
        names: List[str] = []
        positions: List[webcolors.IntegerRGB] = []

        for hex_, name in hex_names.items():
            names.append(name)
            positions.append(webcolors.hex_to_rgb(hex_))

        return names, positions

    def get_colour(self) -> Union[str, List[str]]:
        """
        Finding color using nearest neighborhood algorithm
        """
        # lets populate some names into spatial name database
        names, positions = self._get_colour()

        kd_tree = KDTree(positions)

        # if mass
        if isinstance(self.rgb, list):
            colour_names: List[str] = []
            for item in self.rgb:
                dist, index = kd_tree.query(item)
                log.info(f'The color {self.rgb} is closest to {names[index]}.')
                colour_names.append(names[index])

                return colour_names

        else:
            # query nearest point
            dist, index = kd_tree.query(self.rgb)
            log.info(f'The color {self.rgb} is closest to {names[index]}.')
            return names[index]

    def make_histogram(self, all_colors: bool = False) -> NoReturn:
        """
        Counting the number of pixels in each cluster
        :param:  all_colors - returning all color in dominant order ( from most to less )
        :return: NoReturn
        """
        num_labels = np.arange(0, len(np.unique(self.clusters.labels_)) + 1)
        hist, _ = np.histogram(self.clusters.labels_, bins=num_labels)
        hist = hist.astype('float32')
        hist /= hist.sum()

        # sort most-common first
        combined = zip(hist, self.clusters.cluster_centers_)
        combined = sorted(combined, key=lambda x: x[0], reverse=True)

        # output the colors in order
        if all_colors:
            hsv_values: List[Tuple] = []
            rgb_values: List[Tuple] = []
            for rows in combined:
                rgb, hsv = self.make_bar(100, 100, rows[1])
                log.info(f'RGB values: {rgb}')
                log.info(f'HSV values: {hsv}')
                hsv_values.append(hsv)
                rgb_values.append(rgb)

            self.rgb, self.hsv = rgb_values, hsv_values

        else:
            rgb, hsv = self.make_bar(100, 100, combined[0][1])  # dominant color
            self.rgb, self.hsv = rgb, hsv

    @staticmethod
    def make_bar(height: int, width: int, color: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Create an image of a given color
        :param: width of the image
        :param: BGR pixel values of the color
        :return: tuple of bar, rgb values, and hsv values
        """
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]

        return (red, green, blue), (hue, sat, val)
