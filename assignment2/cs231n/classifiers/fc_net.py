from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
    self,
    hidden_dims,
    input_dim=3 * 32 * 32,
    num_classes=10,
    dropout_keep_ratio=1,
    normalization=None,
    reg=0.0,
    weight_scale=1e-2,
    dtype=np.float32,
    seed=None,
    ):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize weights and biases
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f"W{i+1}"] = np.random.normal(0, weight_scale, (dims[i], dims[i+1]))
            self.params[f"b{i+1}"] = np.zeros(dims[i+1])

            # Initialize batch/layer normalization parameters
            if self.normalization in ["batchnorm", "layernorm"] and i < self.num_layers - 1:
                self.params[f"gamma{i+1}"] = np.ones(dims[i+1])
                self.params[f"beta{i+1}"] = np.zeros(dims[i+1])

        # Dropout parameters
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # Batch/layer normalization parameters
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for _ in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        # Cast parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for dropout and batch/layer normalization
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        # Forward pass
        scores = None
        cache = {}
        out = X
        for i in range(self.num_layers - 1):
            W = self.params[f"W{i+1}"]
            b = self.params[f"b{i+1}"]
            out, cache[f"affine{i+1}"] = affine_forward(out, W, b)

            # Batch/layer normalization
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i+1}"]
                beta = self.params[f"beta{i+1}"]
                out, cache[f"norm{i+1}"] = batchnorm_forward(out, gamma, beta, self.bn_params[i])
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i+1}"]
                beta = self.params[f"beta{i+1}"]
                out, cache[f"norm{i+1}"] = layernorm_forward(out, gamma, beta, self.bn_params[i])

            # ReLU activation
            out, cache[f"relu{i+1}"] = relu_forward(out)

            # Dropout
            if self.use_dropout:
                out, cache[f"dropout{i+1}"] = dropout_forward(out, self.dropout_param)

        # Final affine layer
        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        scores, cache[f"affine{self.num_layers}"] = affine_forward(out, W, b)

        # If test mode, return scores
        if mode == "test":
            return scores

        # Compute loss and gradients
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization
        for i in range(self.num_layers):
            W = self.params[f"W{i+1}"]
            loss += 0.5 * self.reg * np.sum(W * W)

        # Backward pass
        dout = dscores
        dout, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = affine_backward(dout, cache[f"affine{self.num_layers}"])
        grads[f"W{self.num_layers}"] += self.reg * self.params[f"W{self.num_layers}"]

        for i in range(self.num_layers - 2, -1, -1):
            # Dropout backward
            if self.use_dropout:
                dout = dropout_backward(dout, cache[f"dropout{i+1}"])

            # ReLU backward
            dout = relu_backward(dout, cache[f"relu{i+1}"])

            # Batch/layer normalization backward
            if self.normalization == "batchnorm":
                dout, grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = batchnorm_backward(dout, cache[f"norm{i+1}"])
            elif self.normalization == "layernorm":
                dout, grads[f"gamma{i+1}"], grads[f"beta{i+1}"] = layernorm_backward(dout, cache[f"norm{i+1}"])

            # Affine backward
            dout, grads[f"W{i+1}"], grads[f"b{i+1}"] = affine_backward(dout, cache[f"affine{i+1}"])
            grads[f"W{i+1}"] += self.reg * self.params[f"W{i+1}"]

        return loss, grads