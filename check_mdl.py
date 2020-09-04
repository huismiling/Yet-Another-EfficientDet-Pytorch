#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: xiao
# Created Time : 2020-08-02
# File Name: check_mdl.py
# Description:
"""

import onnx
import onnx.utils
from onnx import helper, optimizer

import onnx_tf.backend as backend
#import caffe2.python.onnx.backend as backend
import numpy as np

# Load the ONNX model
model = onnx.load("convert/efficientdet-d0.onnx")

# Check that the IR is well formed
print(onnx.IR_VERSION)
onnx.checker.check_model(model)
print(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
passes = ['eliminate_deadend',
          'eliminate_identity',
          'eliminate_nop_pad',
          'eliminate_nop_transpose',
          'eliminate_unused_initializer',
          'extract_constant_to_initializer',
          'fuse_add_bias_into_conv',
          #'fuse_bn_into_conv',
          'fuse_consecutive_concats',
          'fuse_consecutive_reduce_unsqueeze',
          'fuse_consecutive_squeezes',
          'fuse_consecutive_transposes',
          'fuse_matmul_add_bias_into_gemm',
          'fuse_pad_into_conv',
          'fuse_transpose_into_gemm'
         ]
model = optimizer.optimize(model, passes)

onnx.checker.check_model(model)
model = onnx.utils.polish_model(model)

model_shapes = onnx.shape_inference.infer_shapes(model)

rep = backend.prepare(model, device="CPU") # or "CPU"
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class caffe2.python.onnx.backend.Workspace)
outputs = rep.run(np.random.randn(1, 3, 512, 512).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print(outputs[0])

