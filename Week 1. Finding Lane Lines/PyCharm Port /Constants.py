import numpy as np

"""test"""
MAGIC_NUMBER = 1

"""Contrast"""
bin_256 = 256
hist_range_min = 0
hist_range_max = 256
masked_array_min = 0
masked_array_max = 255

"""range"""
in_range_min = 250
in_range_max = 255

"""gaussian"""
kernel_size = 1

"""canny"""
low_threshold = 50
high_threshold = 150


"""hough"""
rho_multiplier =  .00000217 #distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold_multiplier = .000043403 #nimum number of votes (intersections in Hough grid cell)
min_line_length_multiplier = .000027127 #minimum number of pixels making up a line
max_line_gap_multiplier = .00000217 #maximum gap in pixels between connectable line segments


"""average lines"""
average_slope_counter = 0
average_left = []
average_right = []

"""good lines"""
good_line = []