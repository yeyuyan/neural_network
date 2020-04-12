# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:08:55 2020

@author: YE677
"""

#test cntk
import cntk

a = [1, 2, 3]
b = [4, 5, 6]

c = cntk.minus(a,b).eval()
print(c)