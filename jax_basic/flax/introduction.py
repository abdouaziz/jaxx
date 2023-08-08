import flax
import flax.linen as nn
from jax.random import PRNGKey
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_dataset():
    train = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    test = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )
    return Dataset(train.data, train.targets), Dataset(test.data, test.targets)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


def accuracy(logits, labels):

    return jnp.mean(jnp.argmax(logits, -1) == labels)


def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["image"])
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(model)
    optimizer = optimizer.apply_gradient(grad)
    loss = cross_entropy_loss(logits, batch["label"])
    acc = accuracy(logits, batch["label"])
    metrics = {"loss": loss, "accuracy": acc}
    return optimizer, metrics


def eval_step(model, batch):
    logits = model(batch["image"])
    loss = cross_entropy_loss(logits, batch["label"])
    return {"loss": loss, "accuracy": accuracy(logits, batch["label"])}


def main():
    rng = PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    train_ds, test_ds = load_dataset()

    train_ds = Dataset(train_ds.x[:1000], train_ds.y[:1000])
    test_ds = Dataset(test_ds.x[:1000], test_ds.y[:1000])

    model = MLP()
    _, initial_params = model.init_by_shape(init_rng, [((1, 28, 28), jnp.float32)])
    optimizer = flax.optim.Adam(learning_rate=1e-3).create(initial_params)

    train_batches = 32
    batch_size = len(train_ds) // train_batches
    train_batches = Dataset(
        np.array_split(train_ds.x, train_batches),
        np.array_split(train_ds.y, train_batches),
    )

    test_batches = 32
    batch_size = len(test_ds) // test_batches
    test_batches = Dataset(
        np.array_split(test_ds.x, test_batches),
        np.array_split(test_ds.y, test_batches),
    )

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in train_batches:
            optimizer, train_metrics = train_step(
                model, optimizer, {"image": batch[0], "label": batch[1]}
            )
            print(f"Training loss: {train_metrics['loss']:.4f}")
        for batch in test_batches:
            test_metrics = eval_step(model, {"image": batch[0], "label": batch[1]})
            print(f"Test loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
