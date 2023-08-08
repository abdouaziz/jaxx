import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints

import optax

from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from tqdm import tqdm

from sklearn.model_selection import train_test_split


class ToxiDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer_name,
        max_length,
    ):

        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comments = self.data["comment_text"].values
        labels = self.data["identity_hate"].values

        comment = comments[idx]
        label = labels[idx]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            return_tensors="jax",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "labels": jnp.array(label, dtype=jnp.int32),
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = jnp.stack(input_ids)
    labels = jnp.stack(labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def data_load(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )


class LinearClassifier(nn.Module):
    input_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.input_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
    




def calculate_loss_acc(model_state, params, batch):
    data_input, labels = batch["input_ids"], batch["labels"]
    # Obtain the logits and predictions of the model for the input data
    logits = model_state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss_acc,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc


def train_epoch(state, train_loader, epoch):
    # Iterate over the train loader
    for batch in tqdm(train_loader, total=len(train_loader)):
        # Perform a train step
        state, loss, acc = train_step(state, batch)
    # Print the loss and accuracy for the epoch
    print(f"Epoch: {epoch+1} | Loss: {loss:.3f} | Accuracy: {acc:.3f}")
    return state


def eval_model(state, eval_loader):
    # Iterate over the evaluation loader
    accs = []
    for batch in tqdm(eval_loader, total=len(eval_loader)):
        # Determine the accuracy
        acc = eval_step(state, batch)
        accs.append(acc)
    # Calculate the mean accuracy over the entire evaluation set
    acc = np.mean(accs)
    print(f"Accuracy: {acc:.3f}")
    return acc


def save_checkpoint(state, path, prefix):
    # Save the state dictionary to the provided path
    checkpoints.save_checkpoint(
        ckpt_dir=path,  # Folder to save checkpoint in
        target=state,  # What to save. To only save parameters, use model_state.params
        step=100,  # Training step or other metric to save best model on
        prefix=prefix,  # Checkpoint file name prefix
        overwrite=True,  # Overwrite existing checkpoint files
    )




def main():
    print("Running .... ")

    model = LinearClassifier(input_dim=128, num_classes=1)

    key = jax.random.PRNGKey(seed=42)

    key, subkey = jax.random.split(key)

    random_weigth = jax.random.normal(subkey, (2, 128))

    params = model.init(key, random_weigth)

    optimizer = optax.adam(learning_rate=1e-3)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )



    dataset = ToxiDataset(
        csv_file="jax_advanced/data/train_data.csv",
        tokenizer_name="bert-base-cased",
        max_length=128,
    )

    train_dataset, eval_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    train_loader = data_load(train_dataset, batch_size=32, shuffle=True)

    eval_loader = data_load(eval_dataset, batch_size=32, shuffle=False)

    num_epochs = 5

    for epoch in range(num_epochs):
        # Perform a training epoch
        model_state = train_epoch(model_state, train_loader, epoch)

        # current state
        model_state = model_state.replace(params=model_state.params)
        # Evaluate the current model
        eval_model(model_state, eval_loader)

        # Save the current model
        save_checkpoint(
            path="jax_advanced/checkpoints/",
            state=model_state,
            prefix="model_",
        )


if __name__ == "__main__":
    main()
