import jax
from jax import numpy as jnp
from jax import random
from jax import jit, grad, vmap

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import flax.linen as nn
from flax.training import train_state

import optax


# Load MNIST dataset

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)
test_data = datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

# Define model


class Model(nn.Module):
    features: nn.Module = nn.Dense(features=256)
    logits: nn.Module = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = nn.relu(self.features(x))
        x = self.logits(x)
        return x


# Define loss function


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


# Define optimizer

optimizer = optax.adam(1e-3)

# Define training step


@jit
def train_step(state, batch):
    def loss_fn(params):
        logits = Model().apply({"params": params}, batch["image"])
        loss = cross_entropy_loss(logits, batch["label"])
        return loss

    grad_fn = jit(grad(loss_fn))
    grad = grad_fn(state.params)
    updates, new_optimizer_state = optimizer.update(grad, state.optimizer_state)
    new_params = optax.apply_updates(state.params, updates)
    return train_state.TrainState.create(
        apply_fn=Model().apply, params=new_params, tx=optimizer
    )


# Define evaluation step


@jit
def eval_step(params, batch):
    logits = Model().apply({"params": params}, batch["image"])
    loss = cross_entropy_loss(logits, batch["label"])
    return loss


# Define training loop


def train():
    state = train_state.TrainState.create(
        apply_fn=Model().apply,
        params=Model().init(random.PRNGKey(0), jnp.ones((1, 784))),
        tx=optimizer,
    )
    for epoch in range(10):
        for batch in train_loader:
            state = train_step(state, batch)
        for batch in test_loader:
            loss = eval_step(state.params, batch)
            print(loss)


train()
