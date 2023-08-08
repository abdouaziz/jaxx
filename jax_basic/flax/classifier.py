import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random
import torch
from torch.utils.data import DataLoader, Dataset
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, transforms
from torchvision import datasets

class Classifier(nn.Module):
    input_size: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.input_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

key = random.PRNGKey(seed=0)

key , init_key , input_key = random.split(key, num=3)

model = Classifier(input_size=28*28, num_classes=10)

params = model.init(init_key, jnp.ones((1, 28*28)))


class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train = datasets.MNIST(
    "data", train=True, transform=transforms.ToTensor()
)
test = datasets.MNIST(
    "data", train=False, transform=transforms.ToTensor()
)



train_images, train_labels, test_images, test_labels =
train_images = train_images.reshape(-1, 28*28).numpy()
train_labels = train_labels.numpy()
test_images = test_images.reshape(-1, 28*28).numpy()
test_labels = test_labels.numpy()

train_dataset = Dataset(train_images, train_labels)
test_dataset = Dataset(test_images, test_labels)

print("hello")