# physinet/train.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from flax.serialization import to_bytes
import optax

from .model import PhysiNetOperator
from .data import load_pde_dataset


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, example_x, lr=1e-3, probabilistic=False):
    model = PhysiNetOperator(probabilistic=probabilistic)
    variables = model.init(rng, example_x)
    params = variables["params"]

    tx = optax.adam(lr)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state


@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        x_in = batch_x[:, -1, :, :, None]
        outputs = state.apply_fn({"params": params}, x_in)

        mean = outputs["mean"][..., 0]
        mse = jnp.mean((mean - batch_y) ** 2)
        return mse, {"mse": mse}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


def main(data_dir="data/synthetic", epochs=10, batch_size=8, seq_len=4, lr=1e-3):

    xs, ys = load_pde_dataset(data_dir, seq_len=seq_len, field_key="u")
    N, T, H, W = xs.shape

    rng = jax.random.PRNGKey(0)
    example_x = jnp.array(xs[0:1, -1, :, :])[..., None]
    state = create_train_state(rng, example_x, lr=lr)

    num_batches = N // batch_size

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        xs_ep = xs[perm]
        ys_ep = ys[perm]

        epoch_loss = 0.0

        for i in range(num_batches):
            bs = slice(i * batch_size, (i + 1) * batch_size)
            batch_x = jnp.array(xs_ep[bs])
            batch_y = jnp.array(ys_ep[bs])

            state, loss, metrics = train_step(state, batch_x, batch_y)
            epoch_loss += float(loss)

        epoch_loss /= max(1, num_batches)
        print(f"[Epoch {epoch+1}/{epochs}] loss={epoch_loss:.6e}")

    print("Saving model → physinet_params.msgpack ...")
    with open("physinet_params.msgpack", "wb") as f:
        f.write(to_bytes(state.params))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
