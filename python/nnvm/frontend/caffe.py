# pylint: disable=import-self, invalid-name, unused-argument
"""Caffe: Caffe Symbol frontend."""
from __future__ import absolute_import as _abs
import tvm
import caffe
import numpy as np
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util
from .common import get_nnvm_op, Renamer, AttrConverter as AttrCvt

__all__ = ['from_caffe']


def _raise_not_supported(attr, op='nnvm'):
    err = "{} is not supported in {}.".format(attr, op)
    raise NotImplementedError(err)


def _conv(inputs, layer, layer_weights):
    kernel = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1
    stride = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1
    pad = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0
    dilation = layer.convolution_param.dilation[0] if len(layer.convolution_param.dilation) else 1
    op_name, attrs = 'conv2d', {}
    attrs['kernel_size'] = (kernel, kernel)
    attrs['strides'] = (stride, stride)
    attrs['padding'] = (pad, pad, pad, pad)
    attrs['dilation'] = (dilation, dilation)
    attrs['channels'] = layer.convolution_param.num_output
    # if layer.convolution_param.has_group():
    attrs['groups'] = layer.convolution_param.group
    conv_layer_weights = layer_weights.get(layer.name, None)
    if len(conv_layer_weights) == 1:
        use_bias = False
    elif len(conv_layer_weights) == 2:
        use_bias = True
    else:
        _raise_not_supported('len(layer_params) >= 3', op_name)
    attrs['use_bias'] = use_bias
    return get_nnvm_op(op_name)(*inputs, **attrs)


def _relu(inputs, layer, layer_weights):
    op_name, attrs = 'relu', {} #TODO(wwcai): prelu
    # for weight_name in layer_weights[layer.name]:
    #     print('deleting relu weights: %s', weight_name)
    #     del inputs[weight_name]
    return get_nnvm_op(op_name)(*inputs, **attrs)


def _pooling(inputs, layer, layer_weights):
    pooling_param = layer.pooling_param
    kernel = pooling_param.kernel_size
    stride = pooling_param.stride
    pad = pooling_param.pad
    pool_type = 'max' if pooling_param.pool == 0 else ''
    pool_type = 'avg' if pooling_param.pool == 1 else pool_type
    if pool_type not in ['avg', 'max']:
        _raise_not_supported('non-avg/max', 'pool2d')
    #TODO(wwcai): global?
    global_pool = 'global' if False else ''
    op_name, new_attrs = '_'.join([global_pool, pool_type, 'pool2d']).strip('_'), {}
    # new_attrs['layout'] = 'NCHW'
    if not global_pool:
        new_attrs['pool_size'] = (kernel, kernel)
        new_attrs['strides'] = (stride, stride)
        new_attrs['padding'] = (pad, pad)
        #new_attrs['ceil_mode'] = (attrs.get('pooling_convention', 'valid') == 'full')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)


def _prelu(inputs, layer, layer_weights):
    op_name, new_attrs = 'prelu', {}
    return get_nnvm_op(op_name)(*inputs, **new_attrs)


def _eltwise(inputs, layer, layer_weights):
    assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
    eltwise_param = layer.eltwise_param
    eltwise_op = ['elemwise_prod', 'elemwise_add', 'elemwise_max'] #TODO(wwcai)
    op_name = eltwise_op[eltwise_param.operation]
    axis = 0 #TODO(wwcai
    #TODO(wwcai)
    conv_ops = ["conv2d", "conv2d_transpose"]
    if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
        # TODO(zhreshold): remove hard coded infershape
        inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
    return get_nnvm_op(op_name)(*inputs)


def _inner_product(inputs, layer, layer_weights):
    ip_param = layer.inner_product_param
    op_name, new_attrs = 'dense', {}
    new_attrs['units'] = ip_param.num_output
    inputs[0] = _sym.flatten(inputs[0])
    # inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
    return get_nnvm_op(op_name)(*inputs, **new_attrs)


def _softmax(inputs, layer, layer_weights):
    ip_param = layer.inner_product_param
    op_name, new_attrs = 'softmax', {}
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'Convolution'       : _conv,
    'ReLU'              : _relu,
    'Pooling'           : _pooling,
    # 'BatchNorm'       : _batch_norm,
    # 'Scale'   : _scale
    # 'ReLU'    : _relu
    'PReLU'             : _prelu,
    'Eltwise'           : _eltwise,
    'InnerProduct'      : _inner_product,
    'Softmax'           : _softmax,
    # 'Dropout':      _dropout,
}


def _convert_operator(self, op_name, inputs, attrs, identity_list=None, convert_map=None):
    """Convert from caffe layer to nnvm operator.
    The converter must specify conversions explicity for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, InnerProduct
    inputs : list of nnvm.Symbol
        List of input symbols.
    attrs : dict
        Dict of operator attributes
    identity_list : list
        List of operators that don't require conversion
    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to nnvm, callable are functions which
        take attrs and return (new_op_name, new_attrs)

    Returns
    -------
    sym : nnvm.Symbol
        Converted nnvm Symbol
    """
    identity_list = identity_list if identity_list else _identity_list
    convert_map = convert_map if convert_map else _convert_map
    if op_name in identity_list:
        sym = get_nnvm_op(op_name)(*inputs, **attrs)
    elif op_name in convert_map:
        sym = convert_map[op_name](inputs, attrs, self._params)
    else:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    return sym


def from_caffe(parsible_net, net):
    """Load caffe net in to nnvm graph.

    Parameters
    ----------
    parsible_net :

    net :

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    weights : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    nodes = {}
    weights = {}
    layer_weight_names = {}

    print('processing weights')
    for layer_name, layer_weights in net.params.iteritems():
        layer_weight_names[layer_name] = []
        for i in range(len(layer_weights)):
            #TODO(wwcai): to remove
            if layer_name.find('relu') == 0:
                continue
            weight_name = (layer_name + '_weight_{}').format(i)
            weights[weight_name] = tvm.nd.array(layer_weights[i].data)
            nodes[weight_name] = _sym.Variable(
                name=weight_name, shape=layer_weights[i].data.shape)
            layer_weight_names[layer_name].append(weight_name)
            print('\tlayer %s: weight %s shape %s' % (layer_name, weight_name, layer_weights[i].data.shape))

    print('processing layer')
    for layer in parsible_net.layer:
        op_name = layer.type
        print('\tlayer %s: \t%s' % (layer.name, op_name))
        if op_name == "Input":
            nodes[layer.name] = _sym.Variable(name=layer.name)
            continue

        inputs = [nodes[i] for i in layer.bottom]
        if layer_weight_names.has_key(layer.name):
            inputs.extend([nodes[i] for i in layer_weight_names[layer.name]])

        # node_output = _fix_outputs(op_name, node.output)
        # assert len(node_output) == len(op.list_output_names()), (
        #     "Number of output mismatch {} vs {} in {}.".format(
        #         len(node_output), len(op.list_output_names()), op_name))
        # for k, i in zip(list(node_output), range(len(node_output))):
        #     _nodes[k] = op[i]

        op = _convert_map[op_name](inputs, layer, layer_weight_names)
        # TODO(wwcai):multiple outputs
        nodes[layer.name] = op

    # now return the outputs
    sym = [nodes[i] for i in net.outputs]
    if len(sym) > 1:
        sym = _sym.Group(sym)
    else:
        sym = sym[0]

    return sym, weights
