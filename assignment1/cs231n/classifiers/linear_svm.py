from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange
#

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # scores for the i-th example
        correct_class_score = scores[y[i]]  # score for the correct class
        for j in range(num_classes):
            if j == y[i]:
                continue  # skip the correct class
            margin = scores[j] - correct_class_score + 1  # delta = 1
            if margin > 0:
                loss += margin
                # Update gradient for incorrect class
                dW[:, j] += X[i]
                # Update gradient for correct class
                dW[:, y[i]] -= X[i]

    # Average the loss and gradient over the number of training examples
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W  # gradient of regularization term

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Compute scores
    scores = X.dot(W)  # shape: (N, C)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)  # shape: (N, 1)

    # Compute margins
    margins = np.maximum(0, scores - correct_class_scores + 1)  # delta = 1
    margins[np.arange(num_train), y] = 0  # ignore the correct class

    # Compute loss
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)  # regularization

    # Compute gradient
    binary = margins
    binary[margins > 0] = 1  # binary mask for positive margins
    row_sum = np.sum(binary, axis=1)  # number of positive margins per example
    binary[np.arange(num_train), y] = -row_sum  # subtract for correct class
    dW = X.T.dot(binary) / num_train  # gradient
    dW += 2 * reg * W  # regularization gradient

    return loss, dW
