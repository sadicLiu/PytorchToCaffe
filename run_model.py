# -*- coding:utf-8 -*-
import caffe
import numpy as np
import os
from PIL import Image


# "ceil_mode: false" with "round_mode: FLOOR".


def inference():
    # init net
    caffe_root = '/home/liuhy/military-object/'
    net_file = caffe_root + 'peleenet.prototxt'
    caffe_model = caffe_root + 'peleenet.caffemodel'
    net = caffe.Net(net_file, caffe_model, caffe.TEST)

    # prepare input
    imgs = os.listdir(caffe_root + 'imgs')
    for a_img in imgs:
        img = Image.open(caffe_root + 'imgs/' + a_img).convert('RGB')
        mean = [0.485, 0.456, 0.406]
        std = [1. / 0.229, 1. / 0.224, 1. / 0.225]
        img = img.resize((224, 224), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= mean
        img *= std

        # inference
        net.blobs['blob1'].data[...] = np.expand_dims(img.transpose((2, 0, 1)), 0)
        out = net.forward()

        # print result
        name_idx = {'soldier': 0, 'warship': 1, 'vehicle': 2, 'tank': 3, 'missile': 4,
                    'warplane': 5, 'rifle': 6, 'submarine': 7, 'cannon': 8}
        idx_name = {v: k for k, v in name_idx.items()}
        print('img:{}, pred:{}'.format(a_img, idx_name[np.argmax(out['fc_blob1'])]))


inference()
