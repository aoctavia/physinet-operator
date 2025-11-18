# # physinet/train.py

# from __future__ import annotations
# import os

# import jax
# import jax.numpy as jnp
# import numpy as np

# from flax.training import train_state
# from flax.serialization import to_bytes
# import optax

# from .model.operator import MultiScaleOperator      # NEW
# from .model.physics import physics_residual         # NEW
# from .data import load_pde_dataset


# # -----------------------------
# # Train State
# # -----------------------------
# class TrainState(train_state.TrainState):
#     batch_stats: dict | None = None


# # -----------------------------
# # Create model + train state
# # -----------------------------
# def create_train_state(rng, example_x, lr=1e-3):
#     model = MultiScaleOperator()      # <<— NEW OPERATOR
#     variables = model.init(rng, example_x)
#     params = variables["params"]

#     tx = optax.adam(lr)

#     state = TrainState.create(
#         apply_fn=model.apply,
#         params=params,
#         tx=tx,
#         batch_stats=None,
#     )
#     return state


# # -----------------------------
# # JIT TRAIN STEP
# # -----------------------------
# @jax.jit
# def train_step(state, batch_x, batch_y, dt=0.1, c=1.0, lambda_phys=1.0):
#     """
#     batch_x: [B, seq_len, H, W]
#     batch_y: [B, H, W]
#     """

#     def loss_fn(params):
#         # Input: last frame of sequence
#         x_in = batch_x[:, -1, :, :][..., None]   # [B,H,W,1]

#         # Model prediction
#         pred_next = state.apply_fn({"params": params}, x_in)  # [B,H,W,1]
#         pred_next = pred_next[..., 0]

#         # Data loss
#         mse_loss = jnp.mean((pred_next - batch_y) ** 2)

#         # Physics-informed loss
#         u_prev = batch_x[:, -3, :, :]
#         u_cur = batch_x[:, -2, :, :]

#         phys = physics_residual(u_prev, u_cur, pred_next, c=c, dt=dt)

#         total = mse_loss + lambda_phys * phys

#         return total, {
#             "mse": mse_loss,
#             "phys": phys,
#             "total": total,
#         }

#     (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     state = state.apply_gradients(grads=grads)
#     return state, loss, metrics


# # -----------------------------
# # MAIN TRAINING FUNCTION
# # -----------------------------
# def main(
#     data_dir="data/synthetic",
#     epochs=10,
#     batch_size=8,
#     seq_len=4,
#     lr=1e-3,
#     dt=0.1,
#     c=1.0,
#     lambda_phys=1.0,
# ):

#     xs, ys = load_pde_dataset(data_dir, seq_len=seq_len, field_key="u")
#     N, T, H, W = xs.shape

#     rng = jax.random.PRNGKey(0)

#     # Example input for initialization
#     example_x = jnp.array(xs[0:1, -1, :, :])[..., None]

#     # Train state
#     state = create_train_state(rng, example_x, lr=lr)

#     num_batches = N // batch_size

#     # Training loop
#     for epoch in range(epochs):

#         perm = np.random.permutation(N)
#         xs_ep = xs[perm]
#         ys_ep = ys[perm]

#         epoch_loss = 0.0

#         for i in range(num_batches):
#             bs = slice(i * batch_size, (i + 1) * batch_size)
#             batch_x = jnp.array(xs_ep[bs])   # [B,seq,H,W]
#             batch_y = jnp.array(ys_ep[bs])   # [B,H,W]

#             state, loss, metrics = train_step(
#                 state,
#                 batch_x,
#                 batch_y,
#                 dt=dt,
#                 c=c,
#                 lambda_phys=lambda_phys,
#             )

#             epoch_loss += float(loss)

#         epoch_loss /= max(1, num_batches)
#         print(
#             f"[Epoch {epoch+1}/{epochs}] total={epoch_loss:.6e} "
#             f"(mse={metrics['mse']:.3e}, phys={metrics['phys']:.3e})"
#         )

#     # Save parameters
#     print("Saving model → physinet_params.msgpack ...")
#     with open("physinet_params.msgpack", "wb") as f:
#         f.write(to_bytes(state.params))


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="data/synthetic")
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--seq_len", type=int, default=4)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--dt", type=float, default=0.1)
#     parser.add_argument("--c", type=float, default=1.0)
#     parser.add_argument("--lambda_phys", type=float, default=1.0)
#     args = parser.parse_args()

#     main(
#         data_dir=args.data_dir,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         seq_len=args.seq_len,
#         lr=args.lr,
#         dt=args.dt,
#         c=args.c,
#         lambda_phys=args.lambda_phys,
#     )
# physinet/train.py

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

from flax.training import train_state
from flax.serialization import to_bytes
import optax

# NEW imports
from .model.operator import MultiScaleOperator
from .model.physics import wave_residual
from .data import load_pde_dataset


# ---------------------------------------------------------
# Train state (standard Flax)
# ---------------------------------------------------------
class TrainState(train_state.TrainState):
    batch_stats: dict | None = None


# ---------------------------------------------------------
# Initialize model & training state
# ---------------------------------------------------------
def create_train_state(
    rng,
    example_x,
    lr=1e-3,
    probabilistic=False,
):
    """
    example_x: [1, H, W, 1]  --> used for shape inference
    """

    model = MultiScaleOperator(probabilistic=probabilistic)

    variables = model.init(rng, example_x)
    params = variables["params"]

    tx = optax.adam(lr)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=None,
    )
    return state


# ---------------------------------------------------------
# One training step (JIT)
# ---------------------------------------------------------
@jax.jit
def train_step(
    state,
    batch_x,      # [B, seq_len, H, W]
    batch_y,      # [B, H, W]
    dt=0.1,
    c=1.0,
    lambda_phys=1.0,
):
    def loss_fn(params):

        # ----- Model input: last frame -----
        x_in = batch_x[:, -1, :, :][..., None]   # [B,H,W,1]

        # ----- Forward -----
        outputs = state.apply_fn({"params": params}, x_in)
        pred_next = outputs["mean"][..., 0]      # [B,H,W]

        # ----- Data loss -----
        mse = jnp.mean((pred_next - batch_y) ** 2)

        # ----- Physics-informed loss -----
        # required: [t-2], [t-1]
        u_prev = batch_x[:, -3, :, :]
        u_cur = batch_x[:, -2, :, :]

        phys_loss = wave_residual(u_prev, u_cur, pred_next, c=c, dt=dt)

        # ----- Total loss -----
        total = mse + lambda_phys * phys_loss

        metrics = {
            "mse": mse,
            "phys": phys_loss,
            "total": total,
        }
        return total, metrics

    # compute gradient
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # update state
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


# ---------------------------------------------------------
# Main training loop
# ---------------------------------------------------------
def main(
    data_dir="data/synthetic_wave",
    epochs=10,
    batch_size=8,
    seq_len=4,
    lr=1e-3,
    dt=0.1,
    c=1.0,
    lambda_phys=1.0,
):
    """
    Loads dataset and trains MultiScaleOperator using data loss + physics loss.
    """

    # -------------------------
    # Load dataset
    # -------------------------
    xs, ys = load_pde_dataset(data_dir, seq_len=seq_len, field_key="u")
    N, T, H, W = xs.shape

    print(f"Loaded dataset: xs={xs.shape}, ys={ys.shape}")

    # -------------------------
    # Init model
    # -------------------------
    rng = jax.random.PRNGKey(0)
    example_x = jnp.array(xs[0:1, -1, :, :])[..., None]   # shape [1,H,W,1]

    state = create_train_state(
        rng,
        example_x,
        lr=lr,
        probabilistic=False,
    )

    # -------------------------
    # Training loop
    # -------------------------
    num_batches = N // batch_size

    for epoch in range(1, epochs + 1):
        # shuffle dataset
        perm = np.random.permutation(N)
        xs_ep = xs[perm]
        ys_ep = ys[perm]

        epoch_loss = 0.0

        for b in range(num_batches):
            bs = slice(b * batch_size, (b + 1) * batch_size)

            batch_x = jnp.array(xs_ep[bs])   # [B,seq,H,W]
            batch_y = jnp.array(ys_ep[bs])   # [B,H,W]

            state, loss, metrics = train_step(
                state,
                batch_x,
                batch_y,
                dt=dt,
                c=c,
                lambda_phys=lambda_phys,
            )

            epoch_loss += float(loss)

        epoch_loss /= max(1, num_batches)
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"total={epoch_loss:.6e}  "
            f"(mse={metrics['mse']:.3e}, phys={metrics['phys']:.3e})"
        )

    # -------------------------
    # Save model
    # -------------------------
    out_path = "physinet_params.msgpack"
    print(f"Saving model → {out_path}")

    with open(out_path, "wb") as f:
        f.write(to_bytes(state.params))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic_wave")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=1.0)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        dt=args.dt,
        c=args.c,
        lambda_phys=args.lambda_phys,
    )
