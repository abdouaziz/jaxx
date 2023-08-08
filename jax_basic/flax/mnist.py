import jax 
import jax.numpy as jnp
from jax import random
from jax import jit, grad, vmap

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints

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
    new_optimizer_state, updates = optimizer.update(grad, state.optimizer_state)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        step=state.step + 1, optimizer_state=new_optimizer_state, params=new_params
    )
    return new_state

# Define evaluation step

@jit
def eval_step(params, batch):
    logits = Model().apply({"params": params}, batch["image"])
    loss = cross_entropy_loss(logits, batch["label"])
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])
    return loss, accuracy

# Initialize model

rng = random.PRNGKey(0)
rng, init_rng = random.split(rng)
_, initial_params = Model().init_by_shape(
    init_rng, [((1, 784), jnp.float32)]
)
state = train_state.TrainState.create(
    apply_fn=Model().apply,
    params=initial_params,
    tx=optimizer,
)

# Train model

for epoch in range(3):
    for batch in train_loader:
        state = train_step(state, batch)

    for batch in test_loader:
        loss, accuracy = eval_step(state.params, batch)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

# Save model

checkpoints.save_checkpoint("./model", state, keep=3)

