import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    path1 = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    module = nn.Residual(path1)
    return nn.Sequential(module, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) \
            for i in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_sum = 0
    
    if opt:
        model.train()
        for idx, (X, y) in enumerate(dataloader):
            opt.reset_grad()
            out = model(X)
            loss = loss_func(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.numpy()
            hit += (out.numpy().argmax(1) == y.numpy() ).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, (X, y) in enumerate(dataloader):
            out = model(X)
            loss = loss_func(out, y)
            loss_sum += loss.numpy()
            hit += (out.numpy().argmax(1) == y.numpy() ).sum()
            total += y.shape[0]
            
    return 1 - hit / total, loss_sum / (idx+1), 

        
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
        )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz",
        )
    train_dataloader = ndl.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        )
    test_dataloader = ndl.data.DataLoader(
        test_dataset,
        batch_size,
        )
    
    model = MLPResNet(784, hidden_dim)
    if optimizer is not None:
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        for e in range(epochs):
            train_acc, train_loss = epoch(train_dataloader, model, opt)
            if e == epochs - 1:
                test_acc, test_loss = epoch(test_dataloader, model)
        return (train_acc, train_loss, test_acc, test_loss)

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")