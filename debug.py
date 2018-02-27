from net import Deconvolution2DLayer
from chainer import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as Variable_pt
from chainer import Variable

def compareCandP(ch0,ch1, batch, dshape):
    ksize, stride, pad = 3,2,1

    chainer_dconv = Deconvolution2DLayer(ch0,ch1,ksize,stride,pad)
    pytorch_dconv = nn.ConvTranspose2d(ch0,ch1,ksize,stride,pad,output_padding=1)

    # w = np.random.rand(ch0,ch1,ksize,ksize)
    # b = np.random.rand(ch1)
    w = np.random.randint(0,255,(ch0,ch1,ksize,ksize)).astype('f')
    b = np.random.randint(0, 255, ch1).astype('f')
    chainer_dconv.c.W.data = w
    chainer_dconv.c.b.data = b
    pytorch_dconv.weight.data = torch.FloatTensor(w)
    pytorch_dconv.bias.data = torch.FloatTensor(b)

    # data = np.random.rand(batch, ch0, dshape[0],dshape[1])
    data = np.random.randint(0,255, (batch, ch0, dshape[0],dshape[1])).astype('f')

    c_result = chainer_dconv(Variable(data)).data
    p_result = pytorch_dconv(Variable_pt(torch.FloatTensor(data))).data.numpy()

    diff = c_result - p_result

    print('diff:')
    print(diff)

    print("diff == 0:")
    print(diff == np.zeros(diff.shape).all())

    #print((c_result == p_result).all())