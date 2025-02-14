from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Get shapes
    num_train = X.shape[0]  # Number of training examples
    num_classes = W.shape[1]  # Number of classes

    for i in range(num_train):
        # Compute scores for the i-th example
        scores = X[i].dot(W)
        
        # Shift scores for numerical stability (subtract the max score)
        scores -= np.max(scores)
        
        # Compute the softmax probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        # Compute the cross-entropy loss
        correct_class_prob = probs[y[i]]
        loss += -np.log(correct_class_prob)
        
        # Compute the gradient for the i-th example
        for c in range(num_classes):
            if c == y[i]:
                dW[:, c] += (probs[c] - 1) * X[i]
            else:
                dW[:, c] += probs[c] * X[i]

    # Average the loss and gradient over the training set
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Get shapes
    num_train = X.shape[0]  # Number of training examples

    # Compute scores
    scores = X.dot(W)
    
    # Shift scores for numerical stability (subtract the max score for each example)
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Compute the softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute the cross-entropy loss
    correct_class_probs = probs[np.arange(num_train), y]
    loss = -np.sum(np.log(correct_class_probs)) / num_train
    
    # Add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute the gradient
    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    
    # Add regularization to the gradient
    dW += reg * W

    return loss, dW