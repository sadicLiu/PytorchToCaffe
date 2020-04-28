import torch

import pytorch_to_caffe
from PeleeNet import *

if __name__ == '__main__':
    name = 'peleenet'
    peleenet = PeleeNet(num_classes=9)
    ckpt = torch.load('../assets/ckpt_peleenet.pth', map_location='cpu')
    peleenet.load_state_dict(ckpt['state_dict'])
    peleenet.eval()

    input_tensor = torch.ones(1, 3, 224, 224)
    print(peleenet.forward(input_tensor))
    # input = Variable(input_tensor)

    # convert
    pytorch_to_caffe.trans_net(peleenet, input_tensor, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

    # analyse
    # blob_dict, tracked_layers = pytorch_analyser.analyse(peleenet, input_tensor)
    # pytorch_analyser.save_csv(tracked_layers, '../tmp/analysis.csv')
