import os
import sys
import pathlib
import pickle as pkl
from typing import Any

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks")
matplotlib.rc("text", usetex=True)

from svise.sde_learning import NeuralSDE
from svise.sdeint import solve_sde
from svise.pca import PCA

CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
from train_nsde import MODEL_DIR, set_seeds, initialize_model, get_data, no_grad
from generate_data import DATA_DIR, CylinderFlowData

FIG_DIR = os.path.join(CURR_DIR, "results", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

FILL_COLOR = "#92c5de"
SAMPLE_COLORS = ("#0571b0", "#fccc21", "#f4a582")
LINEWIDTH = 0.5

sns.set(style="ticks")
matplotlib.rc("text", usetex=False)
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
WIDTH_INCHES = 89 / 25.4  # Nature column width


def convert_width(fsize: tuple[float, float]) -> tuple[float, float]:
    width = fsize[0]
    return tuple(size * WIDTH_INCHES / width for size in fsize)


def load_latest_ckpt(nsamples: int, device):
    with open(os.path.join(MODEL_DIR, "init_state_freeze.pkl"), "rb") as f:
        init_state = pkl.load(f)
    init_state["n_reparam_samples"] = nsamples
    nsde = initialize_model(init_state)
    ckpt_path = sorted([ckpt for ckpt in os.listdir(MODEL_DIR) if "nsde" in ckpt])[-1]
    print(f"Loading {ckpt_path}.")
    state_dict = torch.load(os.path.join(MODEL_DIR, ckpt_path), map_location=device)
    nsde.load_state_dict(state_dict)
    # get ready to perform predictions
    nsde.resample_sde_params()
    nsde.eval()
    nsde.to(device)  # Move the model to the specified device
    return nsde


def load_pca(data: dict[str, Any]) -> PCA:
    lin_model = PCA()
    lin_model.load_state_dict(data["lin_model"])
    lin_model.eval()
    return lin_model


def vorticity(dx, dy, vel_x, vel_y):
    """Compute the vorticity field (note this will be incorrect on the boundary)"""
    vort = -(torch.roll(vel_x, -1, dims=0) - torch.roll(vel_x, 1, dims=0)) / (
            2 * dy
    ) + (torch.roll(vel_y, -1, dims=1) - torch.roll(vel_y, 1, dims=1)) / (2 * dx)
    return vort


@no_grad
def get_valid_preds(nsde: NeuralSDE, nsamples: int, data: dict[str, Any], device):
    y0 = data["valid_z"][:1].repeat(nsamples, 1).to(device).to(torch.float64)  # Move y0 to the device
    sde_kwargs = {"adaptive": True, "atol": 1e-2, "rtol": 1e-2, "dt": 1e-2}
    t_eval = data["valid_t"][:40].to(device)  # Move t_eval to the device
    print(f"Solving SDE between {t_eval[0]} and {t_eval[-1]}.")
    ys = solve_sde(nsde, x0=y0, t_eval=t_eval, **sde_kwargs)
    return t_eval, ys


@no_grad
def plot_state_estimate(nsde: NeuralSDE, nsamples: int, data: dict[str, Any], device):
    t = data["t"].to(device)  # Move t to the device
    ys = nsde.marginal_sde.generate_samples(t, nsamples).transpose(1, 0)
    plot_latent_states(t.cpu(), ys.cpu(), data, "z", "state_estimate.png")  # Move tensors back to CPU for plotting


@no_grad
def plot_latent_states(t_eval, ys, data: dict[str, Any], data_key: str, fname: str):
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    alphas = [a + 0.1 for a in alphas]
    percentiles = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]

    n_plots = 10
    num_data = len(t_eval)
    fig, axs = plt.subplots(n_plots, 1, figsize=convert_width((6, int(3 * n_plots / 5))))

    # 新增统计量计算
    ground_truth = data[data_key][:num_data]
    pred_mean = ys.mean(dim=1)  # (T, K)

    # 计算每个维度的指标
    mse_per_dim = torch.mean((pred_mean - ground_truth) ** 2, dim=0)
    mae_per_dim = torch.mean(torch.abs(pred_mean - ground_truth), dim=0)

    # 全局统计量
    global_mse = torch.mean(mse_per_dim).item()
    global_mae = torch.mean(mae_per_dim).item()

    for j in range(n_plots):
        # 绘制原始曲线
        axs[j].plot(t_eval, ground_truth[:, j], color="black", linewidth=LINEWIDTH, label="Ground Truth")

        # 在右侧添加统计量文本
        text_str = f"MSE: {mse_per_dim[j].item():.2e}\nMAE: {mae_per_dim[j].item():.2e}"
        axs[j].text(
            0.95, 0.70, text_str,
            transform=axs[j].transAxes,
            ha="right", va="top",
            fontsize=4,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0)
        )

        axs[j].spines["top"].set_visible(False)
        axs[j].spines["right"].set_visible(False)
        axs[j].set_ylabel(f"$x_{j}$", rotation=0, ha="right", va="center")
        axs[j].set_yticks([])

        if j < n_plots - 1:
            axs[j].tick_params(axis="x", which="both", bottom=False)
            axs[j].spines["bottom"].set_visible(False)
            axs[j].set_xticklabels([])

    # 在最后一个子图添加全局统计量
    # 在最后一个子图添加全局统计量 (调整 y 坐标和对齐方式)
    axs[-1].text(
        0.5,  # x 居中
        -0.9,  # MAE 的 y 坐标（略微上移）
        f"Global MAE: {global_mae:.2e}",
        transform=axs[-1].transAxes,
        ha="center",
        va="top",
        fontsize=6,
        color="#666666",
        linespacing=1.5
    )

    axs[-1].text(
        0.5,  # x 居中
        -1.5,  # MSE 的 y 坐标（下移 0.1）
        f"Global MSE: {global_mse:.2e}",
        transform=axs[-1].transAxes,
        ha="center",
        va="top",
        fontsize=6,
        color="#666666",
        linespacing=4
    )
    # 原有置信区间和样本线绘制保持不变
    for quant, alpha in zip(percentiles, alphas):
        lb = torch.quantile(ys, 1 - quant, dim=1)
        ub = torch.quantile(ys, quant, dim=1)
        for j in range(n_plots):
            lb_, ub_ = lb.T[j], ub.T[j]
            axs[j].fill_between(t_eval, lb_, ub_, alpha=alpha, color=FILL_COLOR)

    for k in range(n_plots):
        for j, color in enumerate(SAMPLE_COLORS):
            axs[k].plot(t_eval, ys[:, j, k], color=color, linewidth=LINEWIDTH, alpha=0.7)

    fname = os.path.join(FIG_DIR, fname)
    fig.savefig(fname, format="pdf", bbox_inches="tight")



def plot_vorticity(ax, vort, data, vmin=None, vmax=None, cmap="Spectral_r"):
    if vmin is None:
        vmin = vort.min()
    if vmax is None:
        vmax = vort.max()
    contourf_kwargs = {
        "cmap": sns.color_palette(cmap, as_cmap=True),
        "levels": 100,
        "extend": "both",
        "vmin": vmin,
        "vmax": vmax,
    }
    # prediction
    mgrid = np.meshgrid(*data["grid"])
    cnf = ax.contourf(*mgrid, vort, **contourf_kwargs)
    ax.axis("equal")
    ax.axis("off")
    circ_kwargs = {
        "fill": True,
        "facecolor": "grey",
        "edgecolor": "k",
        "linewidth": 1.0,
        "zorder": 100,
        "alpha": 1.0,
    }
    ax.add_patch(plt.Circle((0, 0), 0.5, **circ_kwargs))
    return cnf


def make_evenly_spaced(x):
    return np.linspace(x.min(), x.max(), len(x))


def plot_streamlines(ax, u, v, data):
    x_grid = (make_evenly_spaced(xg) for xg in data["grid"])
    mgrid = np.meshgrid(*x_grid)
    stream_set = ax.streamplot(*mgrid, u, v, color="C0", linewidth=0.5, density=4, arrowsize=0)
    stream_set.lines.set_alpha(0.3)


def grid_reshape(y):
    return y.reshape(*CylinderFlowData.adjusted_grid)


def add_colorbar(fig, ax, contour, gap=0.05, height=0.02):
    # Get the position of the plot
    pos = ax.get_position()

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - gap, pos.width, height])

    # Add the colorbar to the new axis
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')

    # Move the colorbar labels to the top side
    cbar.ax.xaxis.set_ticks_position('top')

    return cbar


@no_grad
def plot_final_vorticity(t_eval, ys, data: dict[str, Any], lin_model: PCA):
    # 确保输入数据为 Tensor（模型运算需要）
    if isinstance(ys, np.ndarray):
        ys = torch.from_numpy(ys).to(lin_model.device)

    # 解码预测结果（保持 Tensor）
    pred_samples = lin_model.decode(ys[-1:]).squeeze(0)

    # 转换为 NumPy 用于后续绘图
    pred_samples_np = pred_samples.detach().cpu().numpy()
    pred_vel_x, pred_vel_y = np.split(pred_samples_np, 2, axis=1)
    pred_norm = np.sqrt(pred_vel_x ** 2 + pred_vel_y ** 2)
    pred_mean = pred_norm.mean(axis=0)
    pred_std = pred_norm.std(axis=0)

    # 处理验证数据（保持 Tensor 直到需要转换）
    valid_y = lin_model.encode(data["valid_y"])
    valid_decoded = lin_model.decode(valid_y[-1:])
    test_vel_x, test_vel_y = np.split(valid_decoded.detach().cpu().numpy(), 2, axis=1)
    test_norm = np.sqrt(test_vel_x ** 2 + test_vel_y ** 2).squeeze(0)

    # 确保 grid 数据为 NumPy
    x, y = data["grid"]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

    # 创建画布
    size = convert_width((11.0, 18))
    fig, ax = plt.subplots(3, 1, figsize=(size[1], size[0]))

    # 统一 vmin/vmax（确保输入为 NumPy）
    vmin, vmax = 0.0, grid_reshape(pred_mean).max()

    # 绘制真实值
    plot_vorticity(ax[0], grid_reshape(test_norm), data, vmin=vmin, vmax=vmax)
    plot_streamlines(ax[0],
                     grid_reshape(test_vel_x.squeeze(0)),
                     grid_reshape(test_vel_y.squeeze(0)),
                     data)

    # 绘制预测均值
    cnf1 = plot_vorticity(ax[1], grid_reshape(pred_mean), data, vmin=vmin, vmax=vmax)
    plot_streamlines(ax[1],
                     grid_reshape(pred_vel_x.mean(axis=0)),
                     grid_reshape(pred_vel_y.mean(axis=0)),
                     data)

    # 绘制预测标准差
    cnf2 = plot_vorticity(ax[2], grid_reshape(pred_std), data, cmap="Blues")
    plot_streamlines(ax[2],
                     grid_reshape(pred_vel_x.mean(axis=0)),
                     grid_reshape(pred_vel_y.mean(axis=0)),
                     data)

    # 添加颜色条
    add_colorbar(fig, ax[1], cnf1, gap=0.05)
    add_colorbar(fig, ax[2], cnf2, gap=0.05)

    # 保存图像
    fname = os.path.join(FIG_DIR, "norm.pdf")
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main(reload: bool = False):
    set_seeds(23)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Determine the device
    nsamples = 512
    with open(os.path.join(DATA_DIR, "encoded_data.pkl"), "rb") as handle:
        data = pkl.load(handle)
    nsde = load_latest_ckpt(nsamples, device)
    lin_model = load_pca(data)
    latent_state_path = os.path.join(CURR_DIR, "latent_state_preds.pkl")
    if reload:
        preds = get_valid_preds(nsde, nsamples, data, device)
        with open(latent_state_path, "wb") as handle:
            pkl.dump(preds, handle)
    else:
        with open(latent_state_path, "rb") as handle:
            preds = pkl.load(handle)
    t_eval, ys = preds
    plot_state_estimate(nsde, nsamples, data, device)
    plot_latent_states(t_eval.cpu(), ys.cpu(), data, "valid_z",
                       "latent_state_preds_single_layer.pdf")  # Move tensors back to CPU for plotting
    plot_final_vorticity(t_eval.cpu(), ys.cpu(), data, lin_model)  # Move tensors back to CPU for plotting


if __name__ == "__main__":
    main(reload=True)