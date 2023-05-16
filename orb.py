# -*- coding: utf-8 -*-
import cv2
import numpy as np


def orb_features(n):
    """
    ORB特征提取
    """
    print('reading images ...')

    descriptors = []
    orb = cv2.ORB_create(10000) #表示提取一幅图像的n个特征, 如需生成字典文件，改为10000

    for i in range(n):
        #print('frames/frame_%04d.png'%(i+1))
        img = cv2.imread('frames/frame_%04d.png'%(i+1))
        keypoint, descriptor = orb.detectAndCompute(img, None)
        #特征转int型
        # print descriptor
        descriptor = descriptor.astype(int)
        #特征转列表
        descriptor = descriptor.tolist()
        descriptors.append(descriptor)

    return descriptors


def update_image(n):
    orb = cv2.ORB_create(500)

    img = cv2.imread('frames/frame_%04d.png'%(n+1))
    keypoint, descriptor = orb.detectAndCompute(img, None)
    descriptor = descriptor.astype(int)
    descriptor = descriptor.tolist()
    return descriptor

