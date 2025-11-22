"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(label_filename, 'rb') as lbl_f:
        magic, num = struct.unpack(">II", lbl_f.read(8))
        labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)
    with gzip.open(image_filename, 'rb') as img_f:
        magic, num, rows, cols = struct.unpack(">IIII", img_f.read(16))
        images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(num, rows*cols).astype(np.float32) / 255.0
    return images, labels


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    B, C = Z.shape

    # 1) 计算 e^{z}
    exps = ndl.ops.exp(Z)                          # (B, C)

    # 2) 对每个样本求和，再 broadcast 回 (B, C)
    sums = exps.sum(axes=1)                 # (B,)
    # 先 reshape 成 (B, 1)，再 broadcast 成 (B, C)
    sums_bc = sums.reshape((B, 1)).broadcast_to((B, C))
    softmax = exps / sums_bc
    # one-hot encode labels
    # select correct class probability
    correct_probs = (softmax * y).sum(axes=1)     # (B,)

    # compute negative log likelihood
    loss = -ndl.ops.log(correct_probs).sum() / B

    return loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_step = int(np.ceil(num_examples / batch))
    for step in range(num_step):
        start = step * batch
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        Z = X_batch @ theta
        safe_max = Z.max(axis=1, keepdims=True)
        Z_exp = np.exp(Z - safe_max)
        Z_exp_sum = Z_exp.sum(axis=1, keepdims=True)
        softmax = Z_exp / Z_exp_sum

        softmax[np.arange(y_batch.shape[0]), y_batch] -= 1
        grad = X_batch.T @ softmax / y_batch.shape[0]

        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network:
         logits = ReLU(X * W1) * W2
    No bias terms, no dropout, no random shuffling.
    Everything done in NumPy.
    """

    iterations = (y.size + batch - 1) // batch
    for i in range(iterations):
        x = ndl.Tensor(X[i * batch : (i+1) * batch, :])
        Z = ndl.relu(x.matmul(W1)).matmul(W2)
        yy = y[i * batch : (i+1) * batch]
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), yy] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        W1_data = W1.realize_cached_data()
        W2_data = W2.realize_cached_data()
        W1_data -= lr * W1.grad.realize_cached_data()
        W2_data -= lr * W2.grad.realize_cached_data()
    return W1, W2



def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
