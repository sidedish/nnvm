"""
Compile Caffe Models
===================
"""

import nnvm
import tvm
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import logging


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)
        urllib.urlretrieve(url, path)


caffe_root = '/data/release/ivs_models/models/face_recognition/'
model_def = caffe_root + 'face_deploy_865.prototxt'
model_weights = caffe_root + 'face_train_test_iter_15000_865.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
parsible_net = caffe_pb2.NetParameter()
text_format.Merge(open(model_def).read(), parsible_net)

# print parsible_net.layer
sym, params = nnvm.frontend.from_caffe(parsible_net, net)

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
transformer = caffe.io.Transformer({'data': (1, 3, 112, 96)})
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
img=caffe.io.load_image('cat.png')
img1 = transformer.preprocess('data', img)
x = np.array(img1)[np.newaxis, :, :, :]

######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
print('\nbuilding nnvm graph')
import nnvm.compiler
target = 'llvm'
shape_dict = {'data': x.shape}
graph, libmod, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
dtype = 'float32'
print('\ncreating module')
m = graph_runtime.create(graph, libmod, ctx)
# set inputs
print('\nsetting input data')
m.set_input('data', tvm.nd.array(x.astype(dtype)))
print('\nsetting weight data')
m.set_input(**params)
# execute
print('\nrunning module')
m.run()
# get outputs
output_shape = (1, 865)
print('\noutputing')
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
print(tvm_output)
