import torch
from image_classification_cnn.model import SimpleCNN


def test_mnist_forward_shape():
    model = SimpleCNN(in_ch=1, n_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_cifar10_forward_shape():
    model = SimpleCNN(in_ch=3, n_classes=10, img_size=32)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_shape_for_arbitrary_in_ch_and_img_size():
    """Regression guard: flat_dim used to be inferred from in_ch alone
    (in_ch==1 -> 7x7, else 8x8), which silently broke for any (in_ch,
    img_size) combination outside MNIST/CIFAR10. It's now computed from
    an actual forward pass, so this in_ch=3/img_size=28 case -- which the
    old heuristic would have gotten wrong -- must also work."""
    model = SimpleCNN(in_ch=3, n_classes=5, img_size=28)
    x = torch.randn(2, 3, 28, 28)
    out = model(x)
    assert out.shape == (2, 5)


def test_custom_class_count():
    model = SimpleCNN(in_ch=1, n_classes=4)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 4)
