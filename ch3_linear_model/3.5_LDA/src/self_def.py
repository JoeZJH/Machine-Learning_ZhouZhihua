# -*- coding: utf-8 -*

'''''
Create on 2017/3/21

@author PY131
'''''

import numpy as np

'''
get the projective point(2D) of a point to a line

@param point: the coordinate of the point form as [a,b]
@param line: the line parameters form as [k, t] which means y = k*x + t
@return: the coordinate of the projective point  
'''
def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    # line[1]为一个数而不是一个点，可以视为点的x轴坐标为0，y轴坐标为line[1]
    # y = kx + t就是这条直线的值
    # （但是其实这个是不科学的，有些直线与x轴无交点，此时无法通过这个方式表示）
    t = line[1]

    if   k == 0:      return [a, t]
    elif k == np.inf: return [0, b]
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]

         
    
    
    