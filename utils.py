import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
#from skimage.transform import resize
#from models import HiFi1Edge
import skimage.io as io
import time
import skimage
import warnings
from PIL import Image

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))

        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    import chainercv
    from chainercv.links import VGG16
    cv_model = VGG16(pretrained_model='imagenet')
    import torch
    model.conv1_1.weight.data[:] = torch.tensor(cv_model.conv1_1.conv.W.data[:, ::-1].copy())
    model.conv1_1.bias.data[:] = torch.tensor(cv_model.conv1_1.conv.b.data)

    model.conv1_2.weight.data[:] = torch.tensor(cv_model.conv1_2.conv.W.data)
    model.conv1_2.bias.data[:] = torch.tensor(cv_model.conv1_2.conv.b.data)

    model.conv2_1.weight.data[:] = torch.tensor(cv_model.conv2_1.conv.W.data)
    model.conv2_1.bias.data[:] = torch.tensor(cv_model.conv2_1.conv.b.data)

    model.conv2_2.weight.data[:] = torch.tensor(cv_model.conv2_2.conv.W.data)
    model.conv2_2.bias.data[:] = torch.tensor(cv_model.conv2_2.conv.b.data)

    model.conv3_1.weight.data[:] = torch.tensor(cv_model.conv3_1.conv.W.data)
    model.conv3_1.bias.data[:] = torch.tensor(cv_model.conv3_1.conv.b.data)
    model.conv3_2.weight.data[:] = torch.tensor(cv_model.conv3_2.conv.W.data)
    model.conv3_2.bias.data[:] = torch.tensor(cv_model.conv3_2.conv.b.data)
    model.conv3_3.weight.data[:] = torch.tensor(cv_model.conv3_3.conv.W.data)
    model.conv3_3.bias.data[:] = torch.tensor(cv_model.conv3_3.conv.b.data)
    
    model.conv4_1.weight.data[:] = torch.tensor(cv_model.conv4_1.conv.W.data)
    model.conv4_1.bias.data[:] = torch.tensor(cv_model.conv4_1.conv.b.data)
    model.conv4_2.weight.data[:] = torch.tensor(cv_model.conv4_2.conv.W.data)
    model.conv4_2.bias.data[:] = torch.tensor(cv_model.conv4_2.conv.b.data)
    model.conv4_3.weight.data[:] = torch.tensor(cv_model.conv4_3.conv.W.data)
    model.conv4_3.bias.data[:] = torch.tensor(cv_model.conv4_3.conv.b.data)

    model.conv5_1.weight.data[:] = torch.tensor(cv_model.conv5_1.conv.W.data)
    model.conv5_1.bias.data[:] = torch.tensor(cv_model.conv5_1.conv.b.data)
    model.conv5_2.weight.data[:] = torch.tensor(cv_model.conv5_2.conv.W.data)
    model.conv5_2.bias.data[:] = torch.tensor(cv_model.conv5_2.conv.b.data)
    model.conv5_3.weight.data[:] = torch.tensor(cv_model.conv5_3.conv.W.data)
    model.conv5_3.bias.data[:] = torch.tensor(cv_model.conv5_3.conv.b.data)
    print('done')
    
    # vgg16 = sio.loadmat(vggmodel)
    # torch_params =  model.state_dict()
    # for k in vgg16.keys():
    #    name_par = k.split('-')
    #    size = len(name_par)
    #     if size  == 2:
    #         name_space = name_par[0] + '.' + name_par[1]
    #         data = np.squeeze(vgg16[k])
    #         torch_params[name_space] = torch.from_numpy(data)
    # model.load_state_dict(torch_params)

def load_vgg16pretrain_half(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            shape = data.shape
            index = int(shape[0]/2)
            if len(shape) == 1:
                data = data[:index]
            else:
                data = data[:index,:,:,:]
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def load_fsds_caffe(model, fsdsmodel='caffe-fsds.mat'):
    fsds = sio.loadmat(fsdsmodel)
    torch_params =  model.state_dict()
    for k in fsds.keys():
        name_par = k.split('-')
        #print (name_par)
        size = len(name_par)

        data = np.squeeze(fsds[k])


        if 'upsample' in name_par:
           # print('skip upsample')
            continue 


        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(fsds[k])
            if data.ndim==2:
                data = np.reshape(data, (data.shape[0], data.shape[1]))

            torch_params[name_space] = torch.from_numpy(data)

        if size  == 3:
           # if 'bias' in name_par:
            #    continue

            name_space = name_par[0] + '_' + name_par[1]+ '.' + name_par[2]
            data = np.squeeze(fsds[k])
           # print(data.shape)
            if data.ndim==2:
               # print (data.shape[0])
                data = np.reshape(data,(data.shape[0], data.shape[1]))
            if data.ndim==1 :                
                data = np.reshape(data, (1, len(data), 1, 1))
            if data.ndim==0:
                data = np.reshape(data, (1))

            torch_params[name_space] = torch.from_numpy(data)

        if size == 4:
           # if 'bias' in name_par:
            #    continue
            data = np.squeeze(fsds[k])
            name_space = name_par[0] + '_' + name_par[1] + name_par[2] + '.' + name_par[3]
            if data.ndim==2:
                data = np.reshape(data,(data.shape[0], data.shape[1], 1, 1))

            torch_params[name_space] = torch.from_numpy(data)

    model.load_state_dict(torch_params)
    print('loaded')


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1,4,1,1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()
