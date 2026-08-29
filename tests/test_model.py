import torch
from model import SimpleCNN


def test_mnist_forward_shape():
    model = SimpleCNN(in_ch=1, n_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_cifar10_forward_shape():
    model = SimpleCNN(in_ch=3, n_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_custom_class_count():
    model = SimpleCNN(in_ch=1, n_classes=4)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 4)
