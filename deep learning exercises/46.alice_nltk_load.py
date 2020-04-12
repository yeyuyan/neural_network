# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:00:57 2020

@author: YE677
"""

import nltk
import ssl

# 取消SSl认证
ssl._create_default_https_context = ssl._create_unverified_context

# 下载nltk数据包
nltk.download()