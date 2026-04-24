"""
q_landscape_compare.py
======================
Load two trained critic checkpoints (beta=0 and beta=0.001, epoch 500)
and produce a Q(s, a_data) heatmap comparison figure.
"""

import argparse
import functools
import math
import os
from collections import defaultdict
from typing import Dict, Tuple

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Maze configs

_UMAZE_CFG: Dict = {
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


def _draw_maze_walls(ax, cfg: Dict) -> None:
    maze_np = np.rot90(np.array(cfg["maze_array"]), k=1)
    cell_size = cfg["cell_size"]
    offset    = cfg["offset"]
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


# Critic model (same architecture as rebrac.py)

def _pytorch_init(fan_in: float):
    bound = math.sqrt(1.0 / fan_in)
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
    return _init


def _uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
    return _init


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        x = jnp.hstack([state, action])
        x = nn.Dense(h_d, kernel_init=_pytorch_init(s_d + a_d),
                     bias_init=nn.initializers.constant(0.1))(x)
        x = nn.relu(x)
        if self.layernorm:
            x = nn.LayerNorm()(x)
        for _ in range(self.n_hiddens - 1):
            x = nn.Dense(h_d, kernel_init=_pytorch_init(h_d),
                         bias_init=nn.initializers.constant(0.1))(x)
            x = nn.relu(x)
            if self.layernorm:
                x = nn.LayerNorm()(x)
        x = nn.Dense(1, kernel_init=_uniform_init(3e-3),
                     bias_init=_uniform_init(3e-3))(x)
        return x.squeeze(-1)


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


def _load_critic_ckpt(ckpt_dir: str, hidden_dim: int = 256,
                      num_critics: int = 2, state_dim: int = 4, action_dim: int = 2):
    """Return (apply_fn, params). apply_fn(params, state, action) → (num_critics, batch)"""
    module = EnsembleCritic(hidden_dim=hidden_dim, num_critics=num_critics)
    dummy_s = jnp.zeros((1, state_dim))
    dummy_a = jnp.zeros((1, action_dim))
    template = module.init(jax.random.PRNGKey(0), dummy_s, dummy_a)

    with open(os.path.join(ckpt_dir, "critic_params.msgpack"), "rb") as f:
        params = flax.serialization.from_bytes(template, f.read())
    return module.apply, params


# Q_data map computation

@functools.partial(jax.jit, static_argnums=(0,))
def _jit_critic_apply(apply_fn, params, states, actions):
    return apply_fn(params, states, actions).min(0)


def compute_q_data_map(
    critic_tuple: Tuple,
    dataset: Dict[str, np.ndarray],
    resolution: int = 50,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Returns (resolution, resolution) Q(s, a_data) map. NaN where no data.
    critic_tuple: (apply_fn, params)
    """
    apply_fn, params = critic_tuple

    cfg = _UMAZE_CFG
    offset, cell_size = cfg["offset"], cfg["cell_size"]
    maze_np = np.rot90(np.array(cfg["maze_array"]), k=1)
    n_rows_phys, n_cols_phys = maze_np.shape
    p_min  = -offset
    p_xmax = n_cols_phys * cell_size - offset
    p_ymax = n_rows_phys * cell_size - offset

    all_states  = dataset["states"]
    all_actions = dataset["actions"]
    N = len(all_states)

    grid_q_data: Dict = defaultdict(list)
    for i in range(0, N, chunk_size):
        s_np = all_states[i : i + chunk_size]
        a_np = all_actions[i : i + chunk_size]
        qd = np.array(_jit_critic_apply(apply_fn, params, jnp.array(s_np), jnp.array(a_np)))
        xs, ys = s_np[:, 0], s_np[:, 1]
        xi = np.clip(((xs - p_min) / (p_xmax - p_min) * resolution).astype(int), 0, resolution - 1)
        yi = np.clip(((ys - p_min) / (p_ymax - p_min) * resolution).astype(int), 0, resolution - 1)
        for j in range(len(s_np)):
            grid_q_data[(xi[j], yi[j])].append(qd[j])

    Q_data_map = np.full((resolution, resolution), np.nan, dtype=np.float32)
    for (xi, yi), vals in grid_q_data.items():
        Q_data_map[yi, xi] = np.mean(vals)
    return Q_data_map


# Comparison figure

def draw_comparison_figure(
    qmap0: np.ndarray,
    qmap1: np.ndarray,
    beta0_val: float,
    beta1_val: float,
    output_path: str,
) -> None:
    cfg = _UMAZE_CFG
    offset, cell_size = cfg["offset"], cfg["cell_size"]
    maze_np = np.rot90(np.array(cfg["maze_array"]), k=1)
    n_rows_phys, n_cols_phys = maze_np.shape
    p_min  = -offset
    p_xmax = n_cols_phys * cell_size - offset
    p_ymax = n_rows_phys * cell_size - offset
    extent = [p_min, p_xmax, p_min, p_ymax]

    # Shared colour scale for direct comparison
    valid = np.concatenate([qmap0[~np.isnan(qmap0)], qmap1[~np.isnan(qmap1)]])
    vmin, vmax = (float(valid.min()), float(valid.max())) if len(valid) else (0.0, 1.0)
    vmin, vmax = 35, 95 # Fixed scale for better visual comparison

    cmap_obj = plt.get_cmap("viridis").copy()
    cmap_obj.set_bad(color="white")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(output_path)[0]

    cb_ticks = [float(t) for t in np.linspace(vmin, vmax, 5)]

    # Both figures share identical layout constants so the map axes are
    # exactly the same physical size whether or not a colorbar is present.
    FIGSIZE   = (5.5, 5.0)
    FIG_LEFT   = 0.11
    FIG_RIGHT  = 0.85
    FIG_TOP    = 0.95
    FIG_BOTTOM = 0.10

    for qmap, beta_val, tag, show_cb in [
        (qmap0, beta0_val, "beta0",    False),
        (qmap1, beta1_val, "beta0001", True),
    ]:

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT,
                            top=FIG_TOP,   bottom=FIG_BOTTOM)

        masked = np.ma.masked_invalid(qmap)
        im = ax.imshow(masked, origin="lower", extent=extent,
                       cmap=cmap_obj, aspect="equal", vmin=vmin, vmax=vmax)
        _draw_maze_walls(ax, cfg)

        ticks = [0, 1, 2, 3, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)

        # append_axes is called for BOTH figures so the maze axes shrinks by
        # the same amount, keeping the two mazes identical in physical size.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)

        if show_cb:
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Q value", fontsize=14, labelpad=15)
            cb.set_ticks(cb_ticks)
            cb.ax.tick_params(labelsize=14)
        else:
            cax.set_visible(False)

        png_path = f"{base}_{tag}.png"
        fig.savefig(png_path, dpi=150)
        pdf_path = f"{base}_{tag}.pdf"
        fig.savefig(pdf_path)

        import sys
        if os.path.exists(pdf_path):
            print(f"  ==> [SUCCESS] PDF 存檔成功！")
            print(f"  ==> 絕對路徑: {os.path.abspath(pdf_path)}")
        else:
            print(f"  ==> [ERROR] 檔案未生成，請檢查權限或磁碟空間。")
            
        plt.close(fig)


# CLI

def parse_args():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Compare Q(s, a_data) heatmaps for two E2E beta checkpoints (epoch 500)"
    )
    p.add_argument("--beta0_ckpt",  default=os.path.join(_script_dir, "checkpoints", "beta0"),
                   help="Checkpoint dir for beta=0   (contains critic_params.msgpack)")
    p.add_argument("--beta1_ckpt",  default=os.path.join(_script_dir, "checkpoints", "beta0001"),
                   help="Checkpoint dir for beta=0.001 (contains critic_params.msgpack)")
    p.add_argument("--beta0_val",   type=float, default=0.0)
    p.add_argument("--beta1_val",   type=float, default=0.001)
    p.add_argument("--dataset",     default="maze2d-umaze-v1")
    p.add_argument("--resolution",  type=int,   default=50)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--num_critics", type=int,   default=2)
    p.add_argument("--output",      default=os.path.join(_script_dir, "outputs", "q_landscape_compare.png"))
    return p.parse_args()


def main():
    args = parse_args()

    print("=== [1/4] Loading dataset ===")
    import d4rl
    import gym
    env = gym.make(args.dataset)
    ds = env.get_dataset()
    dataset = {
        "states":  ds["observations"].astype(np.float32),
        "actions": ds["actions"].astype(np.float32),
    }

    print("=== [2/4] Loading checkpoints ===")
    critic0 = _load_critic_ckpt(args.beta0_ckpt, args.hidden_dim, args.num_critics)
    critic1 = _load_critic_ckpt(args.beta1_ckpt, args.hidden_dim, args.num_critics)

    print("=== [3/4] Computing Q_data maps ===")
    qmap0 = compute_q_data_map(critic0, dataset, args.resolution)
    qmap1 = compute_q_data_map(critic1, dataset, args.resolution)
    print(f"    beta={args.beta0_val}  Q_data range: [{np.nanmin(qmap0):.2f}, {np.nanmax(qmap0):.2f}]")
    print(f"    beta={args.beta1_val}  Q_data range: [{np.nanmin(qmap1):.2f}, {np.nanmax(qmap1):.2f}]")

    print("=== [4/4] Drawing comparison figure ===")
    draw_comparison_figure(
        qmap0, qmap1,
        args.beta0_val, args.beta1_val,
        args.output,
    )


if __name__ == "__main__":
    main()
