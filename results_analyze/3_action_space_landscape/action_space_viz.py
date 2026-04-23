"""
action_space_viz.py
-------------------
Visualize Q(s_anchor, a) landscape and action gradients for three E2EBeta
configurations (beta=0, 0.001, 0.005) at a fixed anchor state in maze2d-umaze-v1.

Output: 4-panel figure
  Panel 0 : Anchor position in UMaze
  Panel 1 : Q landscape  beta = 0
  Panel 2 : Q landscape  beta = 0.001
  Panel 3 : Q landscape  beta = 0.005

"""

import argparse
import functools
import math
import os

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


A4_W, A4_H = 8.27, 2.8

_UMAZE_CFG = {
    "maze_array": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    "cell_size": 1.0,
    "offset": 0.607,
}


def _draw_maze_walls(ax, cfg: dict) -> None:
    maze_np = np.rot90(np.array(cfg["maze_array"]), k=1)
    cell_size, offset = cfg["cell_size"], cfg["offset"]
    n_rows, n_cols = maze_np.shape
    for row in range(n_rows):
        for col in range(n_cols):
            if maze_np[row, col] == 1:
                x0 = col * cell_size - offset
                y0 = (n_rows - 1 - row) * cell_size - offset
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), cell_size, cell_size,
                    linewidth=0, facecolor="gray", alpha=0.5, zorder=2,
                ))


def _draw_maze_panel(ax, obs: np.ndarray, s_anchor: np.ndarray, radius: float) -> None:
    cfg = _UMAZE_CFG
    cell_size, offset = cfg["cell_size"], cfg["offset"]
    maze_np = np.rot90(np.array(cfg["maze_array"]), k=1)
    n_rows, n_cols = maze_np.shape
    p_min  = -offset
    p_xmax = n_cols * cell_size - offset
    p_ymax = n_rows * cell_size - offset

    # dataset trajectories
    step = max(1, len(obs) // 3000)
    ax.scatter(obs[::step, 0], obs[::step, 1],
               s=1, c="steelblue", alpha=0.25, linewidths=0, zorder=1)

    _draw_maze_walls(ax, cfg)

    # radius circle
    ax.add_patch(mpatches.Circle(
        (s_anchor[0], s_anchor[1]), radius,
        fill=False, edgecolor="red", linewidth=1.5, linestyle="--", zorder=4,
    ))

    ax.scatter(s_anchor[0], s_anchor[1], s=10, marker="*", c="red", zorder=5,
               label=(
                   rf"anchor  $s$=[{s_anchor[0]:.2f}, {s_anchor[1]:.2f},"
                   rf" {s_anchor[2]:.2f}, {s_anchor[3]:.2f}]"
               ))

    ax.set_xlim(p_min, p_xmax)
    ax.set_ylim(p_min, p_ymax)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.tick_params(axis='both', which='major', labelsize=4, length=2, width=0.4)
    ax.set_xlabel("x", fontsize=5)
    ax.set_ylabel("y", fontsize=5)
    ax.set_title("Anchor in UMaze", fontsize=6, fontweight="bold")
    ax.legend(loc="upper left", fontsize=3, framealpha=0.6, markerscale=1.0,
              handlelength=0.5, handletextpad=0.4, borderpad=0.2, labelspacing=0.2)


# Model (same architecture as rebrac.py)

def _pytorch_init(fan_in: float):
    bound = math.sqrt(1.0 / fan_in)
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
    return _init

def _uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
    return _init

def _identity(x):
    return x


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        layers = [
            nn.Dense(h_d, kernel_init=_pytorch_init(s_d + a_d),
                     bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else _identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(h_d, kernel_init=_pytorch_init(h_d),
                         bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.LayerNorm() if self.layernorm else _identity,
            ]
        layers.append(
            nn.Dense(1, kernel_init=_uniform_init(3e-3),
                     bias_init=_uniform_init(3e-3))
        )
        net = nn.Sequential(layers)
        return net(jnp.hstack([state, action])).squeeze(-1)


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 2
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None, out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        return ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(state, action)


def load_critic(ckpt_dir: str,
                hidden_dim: int = 256,
                num_critics: int = 2,
                layernorm: bool = True,
                n_hiddens: int = 3):
    """Return (apply_fn, params). apply_fn(params, state, action) → (num_critics, batch)"""
    module = EnsembleCritic(
        hidden_dim=hidden_dim, num_critics=num_critics,
        layernorm=layernorm, n_hiddens=n_hiddens,
    )
    dummy_s = jnp.zeros((1, 4))
    dummy_a = jnp.zeros((1, 2))
    template = module.init(jax.random.PRNGKey(0), dummy_s, dummy_a)

    ckpt_file = os.path.join(ckpt_dir, "critic_params.msgpack")
    with open(ckpt_file, "rb") as f:
        params = flax.serialization.from_bytes(template, f.read())

    return module.apply, params


# Dataset & anchor

def load_d4rl(dataset_name: str):
    import d4rl  # noqa
    import gym
    env = gym.make(dataset_name)
    ds = env.get_dataset()
    obs     = ds["observations"].astype(np.float32)  # (N, 4): x, y, vx, vy
    actions = ds["actions"].astype(np.float32)        # (N, 2): ax, ay
    return obs, actions


def find_anchor(obs, actions, anchor_x, anchor_y, radius):
    """
    Returns:
      s_anchor        : (4,) — dataset point closest to target coords
      support_actions : (M, 2) — actions within L2 radius of s_anchor
    """
    pos    = obs[:, :2]
    target = np.array([anchor_x, anchor_y], dtype=np.float32)
    dists  = np.linalg.norm(pos - target, axis=1)

    idx_anchor = int(np.argmin(dists))
    s_anchor   = obs[idx_anchor].copy()

    dists_from_anchor = np.linalg.norm(pos - s_anchor[:2], axis=1)
    support_actions   = actions[dists_from_anchor < radius]

    print(f"[Anchor] idx={idx_anchor}  s={s_anchor}  nearest_dist={dists[idx_anchor]:.4f}")
    print(f"[Support] {len(support_actions)} samples within L2={radius} "
          f"of s_anchor=({s_anchor[0]:.3f}, {s_anchor[1]:.3f})")
    return s_anchor, support_actions


# Q computation (vmap over action grid)

@functools.partial(jax.jit, static_argnums=(0,))
def _q_and_grad_batch(apply_fn, params, s_anchor_j, action_grid_j):
    def q_single(a):
        return apply_fn(params, s_anchor_j[None], a[None]).min(0).squeeze()
    return jax.vmap(jax.value_and_grad(q_single))(action_grid_j)


def compute_q_landscape(apply_fn, params, s_anchor, grid_res=50):
    """
    Returns:
      q_2d    : (grid_res, grid_res)  row=ay, col=ax
      grad_2d : (grid_res, grid_res, 2)
      ax_1d   : (grid_res,)
      ay_1d   : (grid_res,)
    """
    ax_1d = np.linspace(-1, 1, grid_res, dtype=np.float32)
    ay_1d = np.linspace(-1, 1, grid_res, dtype=np.float32)
    axx, ayy = np.meshgrid(ax_1d, ay_1d)
    grid = np.stack([axx.ravel(), ayy.ravel()], axis=1)  # (N, 2)

    q_vals, grads = _q_and_grad_batch(
        apply_fn, params,
        jnp.array(s_anchor),
        jnp.array(grid),
    )
    q_2d    = np.array(q_vals).reshape(grid_res, grid_res)
    grad_2d = np.array(grads).reshape(grid_res, grid_res, 2)
    return q_2d, grad_2d, ax_1d, ay_1d


# Plotting

def _draw_panel(ax, q_2d, grad_2d, ax_1d, ay_1d,
                support_actions, beta_val, vmin, vmax, radius=0.005, stride=4):
    im = ax.imshow(
        q_2d, origin="lower", extent=[-1, 1, -1, 1],
        cmap="viridis", aspect="equal",
        vmin=vmin, vmax=vmax, zorder=1,
    )

    # Normalised gradient field
    iy_idx = np.arange(0, q_2d.shape[0], stride)
    ix_idx = np.arange(0, q_2d.shape[1], stride)
    iyy, ixx = np.meshgrid(iy_idx, ix_idx, indexing="ij")
    qx = ax_1d[ixx.ravel()]
    qy = ay_1d[iyy.ravel()]
    gu = grad_2d[iyy.ravel(), ixx.ravel(), 0]
    gv = grad_2d[iyy.ravel(), ixx.ravel(), 1]
    mag = np.sqrt(gu ** 2 + gv ** 2) + 1e-8
    ax.quiver(
        qx, qy, gu / mag, gv / mag,
        color="white", alpha=0.80,
        scale=22, width=0.004,
        headwidth=3.5, headlength=4,
        zorder=3,
    )

    # Data support actions (red dots)
    if len(support_actions) > 0:
        ax.scatter(
            support_actions[:, 0], support_actions[:, 1],
            c="red", s=3, linewidths=0, zorder=5,
            label=f"data support (n={len(support_actions)}, r={radius})",
        )
        ax.legend(loc="upper left", fontsize=3,
                  framealpha=0.55, edgecolor="none",
                  handlelength=0.5, handletextpad=0.2, borderpad=0.2)

    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5, zorder=2)
    ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.5, zorder=2)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$a_x$", fontsize=5)
    ax.set_ylabel(r"$a_y$", fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=4, length=2, width=0.4)
    ax.set_title(rf"E2E$\beta$ = {beta_val}", fontsize=6, fontweight="bold")
    return im


def make_figure(q0, g0, q1, g1, q2, g2, ax_1d, ay_1d,
                support_actions, beta0_val, beta1_val, beta2_val,
                s_anchor, output_path, obs=None, radius=0.05):
    vmin0, vmax0 = float(q0.min()), float(q0.max())
    vmin1, vmax1 = float(q1.min()), float(q1.max())
    vmin2, vmax2 = float(q2.min()), float(q2.max())

    print(f"[Q range] beta={beta0_val}: [{vmin0:.2f}, {vmax0:.2f}]")
    print(f"[Q range] beta={beta1_val}: [{vmin1:.2f}, {vmax1:.2f}]")
    print(f"[Q range] beta={beta2_val}: [{vmin2:.2f}, {vmax2:.2f}]")

    fig, axes = plt.subplots(1, 4, figsize=(A4_W, A4_H),
                             gridspec_kw={"width_ratios": [1.1, 1.3, 1.3, 1.3]})

    _draw_maze_panel(axes[0], obs if obs is not None else np.zeros((0, 4)),
                     s_anchor, radius)

    im0 = _draw_panel(axes[1], q0, g0, ax_1d, ay_1d,
                      support_actions, beta0_val, vmin0, vmax0, radius)
    im1 = _draw_panel(axes[2], q1, g1, ax_1d, ay_1d,
                      support_actions, beta1_val, vmin1, vmax1, radius)
    im2 = _draw_panel(axes[3], q2, g2, ax_1d, ay_1d,
                      support_actions, beta2_val, vmin2, vmax2, radius)

    for im, ax_ in zip([im0, im1, im2], axes[1:]):
        cb = fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.02)
        cb.set_label("Q value", fontsize=5)
        cb.ax.tick_params(labelsize=4, length=2, width=0.4)
        for spine in cb.ax.spines.values():
            spine.set_linewidth(0.4)

    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linewidth(0.4)

    plt.subplots_adjust(left=0.05, right=0.93, top=0.93, bottom=0.10, wspace=0.7)

    fig.set_size_inches(A4_W, A4_H)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    print(f"[Output PNG] {output_path}")

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    print(f"[Output PDF] {pdf_path}")

    plt.close(fig)


# CLI

def parse_args():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Action-space Q landscape: compare three E2EBeta checkpoints (maze2d-umaze-v1)"
    )
    p.add_argument("--beta0_ckpt",   default=os.path.join(_script_dir, "checkpoints", "beta0"),
                   help="Checkpoint dir for beta=0   (contains critic_params.msgpack)")
    p.add_argument("--beta1_ckpt",   default=os.path.join(_script_dir, "checkpoints", "beta0001"),
                   help="Checkpoint dir for beta=0.001")
    p.add_argument("--beta2_ckpt",   default=os.path.join(_script_dir, "checkpoints", "beta0005"),
                   help="Checkpoint dir for beta=0.005")
    p.add_argument("--beta0_val",    type=float, default=0.0)
    p.add_argument("--beta1_val",    type=float, default=0.001)
    p.add_argument("--beta2_val",    type=float, default=0.005)
    p.add_argument("--anchor_x",     type=float, default=1.0)
    p.add_argument("--anchor_y",     type=float, default=1.0)
    p.add_argument("--radius",       type=float, default=0.005)
    p.add_argument("--grid_res",     type=int,   default=50)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--num_critics",  type=int,   default=2)
    p.add_argument("--output",       default=os.path.join(_script_dir, "outputs", "action_space_compare.png"))
    return p.parse_args()


def main():
    args = parse_args()

    print("=== [1/4] Loading dataset ===")
    obs, actions = load_d4rl("maze2d-umaze-v1")
    s_anchor, support_actions = find_anchor(
        obs, actions, args.anchor_x, args.anchor_y, args.radius
    )

    print("=== [2/4] Loading checkpoints ===")
    apply0, params0 = load_critic(args.beta0_ckpt, args.hidden_dim, args.num_critics)
    apply1, params1 = load_critic(args.beta1_ckpt, args.hidden_dim, args.num_critics)
    apply2, params2 = load_critic(args.beta2_ckpt, args.hidden_dim, args.num_critics)

    print("=== [3/4] Computing Q landscapes ===")
    q0, g0, ax_1d, ay_1d = compute_q_landscape(apply0, params0, s_anchor, args.grid_res)
    q1, g1, _,    _      = compute_q_landscape(apply1, params1, s_anchor, args.grid_res)
    q2, g2, _,    _      = compute_q_landscape(apply2, params2, s_anchor, args.grid_res)

    print(f"    beta={args.beta0_val}  Q range: [{q0.min():.2f}, {q0.max():.2f}]")
    print(f"    beta={args.beta1_val}  Q range: [{q1.min():.2f}, {q1.max():.2f}]")
    print(f"    beta={args.beta2_val}  Q range: [{q2.min():.2f}, {q2.max():.2f}]")

    print("=== [4/4] Plotting ===")
    make_figure(
        q0, g0, q1, g1, q2, g2, ax_1d, ay_1d,
        support_actions, args.beta0_val, args.beta1_val, args.beta2_val,
        s_anchor, args.output,
        obs=obs,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()
