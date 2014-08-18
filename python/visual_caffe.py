import numpy as np
import matplotlib.pyplot as plt

import caffe
from caffe import imagenet
import sys
import pdb

# our network takes BGR images, so we need to switch color channels
def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
        if im.shape[2] == 1:
            img = np.ones( (im.shape[0], im.shape[1], 3) )
            img[:,:,0] = im[:,:,0]
            img[:,:,1] = im[:,:,0]
            img[:,:,2] = im[:,:,0]
            im = img
    plt.imshow(im)
    plt.show()

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    showimage(data)


# Make sure that caffe is on the python path:
caffe_root = '../' # this file is expected to be in {caffe_root}/examples

sys.path.insert(0, caffe_root + 'python')


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



# net = caffe.imagenet.ImageNetClassifier('../models2/SED_HS_train.prototxt',
# '../models2/SED_HS_quick_iter_100000')
net = caffe.imagenet.ImageNetClassifier('../models2/SED_test.prototxt',
 '../models2/SED_HS_quick_iter_200000')
net.caffenet.set_phase_test()
net.caffenet.set_mode_cpu()

#scores = net.predict('/media/Backup/Train400PatchWith128/img-10.21op2-p-046t000/1000.jpg')
#print scores
#[(k, v.data.shape) for k, v in net.caffenet.blobs.items()]
#[(k, v[0].data.shape) for k, v in net.caffenet.params.items()]



# index four is the center crop
#image = net.caffenet.blobs['data'].data[4].copy()
#image -= image.min()
#image /= image.max()
#pdb.set_trace()
#showimage(image.transpose(1, 2, 0))

filters = net.caffenet.params['conv1'][0].data#shape (32,3,5,5)
vis_square(filters.transpose(0, 2, 3, 1)) #shape (32,5,5,3)
filters = net.caffenet.params['conv2'][0].data
vis_square(filters[:32].reshape(32**2, 5, 5))
filters = net.caffenet.params['conv3'][0].data
vis_square(filters[:64].reshape(32*64, 5, 5))
# feat = net.caffenet.blobs['conv2'].data[4]
# vis_square(feat, padval=0.5)
