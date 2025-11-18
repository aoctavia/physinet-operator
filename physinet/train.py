from __future__ import annotations
import os
import numpy as np

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.serialization import to_bytes
import optax

# ================================================
#  Import models (folder physinet/models/)
# ================================================
from .models.operator import MultiScaleOperator
from .models.physi_operator import PhysiNetOperator
from .models.physics import wave_residual

# ================================================
#  Import dataset loader
# ================================================
from .data import load_pde_dataset


# ------------------------------------------------
#  Training State
# ------------------------------------------------
class TrainState(train_state.TrainState):
    pass


# ------------------------------------------------
#  Create Model
# ------------------------------------------------
def create_operator(rng, example_x, lr=1e-3, multiscale=False):
    if multiscale:
        model = MultiScaleOperator()
    else:
        model = PhysiNetOperator()

    variables = model.init(rng, example_x)
    params = variables["params"]

    tx = optax.adam(lr)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ------------------------------------------------
#  Loss Function
# ------------------------------------------------
def loss_fn(params, apply_fn, x, y):
    # Forward model
    outputs = apply_fn({"params": params}, x)
    pred = outputs["mean"][..., 0]  # remove channel

    # MSE
    mse = jnp.mean((pred - y) ** 2)

    # Physics loss (wave equation)
    # Dummy residual if seq not provided
    physics = 0.0

    return mse + 0.1 * physics, {"mse": mse, "physics": physics}


# ------------------------------------------------
#  JIT-compiled train step
# ------------------------------------------------
@jax.jit
def train_step(state, batch_x, batch_y):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params,
        state.apply_fn,
        batch_x,
        batch_y,
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


# ------------------------------------------------
#  Main Training Loop
# ------------------------------------------------
def main(data_dir="data/synthetic_wave", epochs=5, batch_size=8, lr=1e-3):

    # Load dataset
    xs, ys = load_pde_dataset(data_dir, seq_len=4, field_key="u")
    N, T, H, W = xs.shape

    # Example input for initializing the model
    example_x = jnp.array(xs[0:1, -1, :, :])[..., None]

    rng = jax.random.PRNGKey(0)

    # MULTISCALE OPERATOR
    state = create_operator(
        rng,
        example_x,
        lr=lr,
        multiscale=True,
    )

    num_batches = N // batch_size

    print(f"Training on {N} samples, {epochs} epochs...")

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        xs_ep = xs[perm]
        ys_ep = ys[perm]

        epoch_loss = 0.0

        for i in range(num_batches):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            batch_x = jnp.array(xs_ep[sl])[:, -1, :, :][..., None]
            batch_y = jnp.array(ys_ep[sl])

            state, loss, metrics = train_step(state, batch_x, batch_y)
            epoch_loss += float(loss)

        epoch_loss /= max(1, num_batches)
        print(f"[Epoch {epoch+1}/{epochs}] loss = {epoch_loss:.6e}")

    # Save
    print("Saving model → physinet_params.msgpack")
    with open("physinet_params.msgpack", "wb") as f:
        f.write(to_bytes(state.params))


# ------------------------------------------------
#  CLI
# ------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/synthetic_wave")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
