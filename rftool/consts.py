'''
random forest tool constants

'''
import os
import numpy as np


out_num = 4
data_num = 8

cv = 3
cv10 = 10

n_jobs_grid = 2

n_jobs = 4

n_trees = 200
n_trees_grid = [25, 50, 100, 200, 400]

data_sep = '\t'

mode, class_col, filt = 'median', 0, .5

type_num = [np.dtype('int'), np.dtype('float')]
type_o = [np.dtype('O')]

# def read(file, header=False):
    # '''
    # ------
    # return: list of list
    # '''
    # data = []
    # with open(file) as f:
        # if header:
            # return f.readline().split(data_sep)
        # else:
            # _ = f.readline()
            # for line in f:
                # line_split = line.split(data_sep)
                # data.append(line_split)
                
    # return data