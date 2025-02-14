from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # Initialize the weights and biases of the two-layer net.                  #
        ############################################################################
        # Initialize weights for the first layer (input to hidden)
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        # Initialize weights for the second layer (hidden to output)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with gradients of the loss with respect to parameters.
        """
        scores = None

        ############################################################################
        # Implement the forward pass for the two-layer net.                        #
        ############################################################################
        # First layer: affine -> ReLU
        W1, b1 = self.params['W1'], self.params['b1']
        out1, cache1 = affine_relu_forward(X, W1, b1)

        # Second layer: affine
        W2, b2 = self.params['W2'], self.params['b2']
        scores, cache2 = affine_forward(out1, W2, b2)

        # If y is None, return scores (test-time forward pass)
        if y is None:
            return scores

        loss, grads = 0, {}

        ############################################################################
        # Implement the backward pass for the two-layer net.                       #
        ############################################################################
        # Compute softmax loss
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to the loss
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backprop through the second layer (affine)
        dout1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)

        # Backprop through the first layer (affine -> ReLU)
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)

        # Add regularization gradients
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads
    
    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign the data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X, where y_pred[i] is the predicted label for X[i].
        """
        # Compute scores using the forward pass
        scores = self.loss(X)  # This calls the forward pass when y is None

        # Predict the class with the highest score
        y_pred = np.argmax(scores, axis=1)

        return y_pred