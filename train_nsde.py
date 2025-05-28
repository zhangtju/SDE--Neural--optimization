"""
Script for training a neural SDE on cylinder flow POD modes.
"""
import os
import sys
import pathlib
import pickle as pkl
from typing import Any
import logging
from functools import wraps
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from svise.sde_learning import NeuralSDE
from svise.sde_learning._single_layer_sde_learner import SingleLayerSDELearner

CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(CURR_DIR)
from generate_data import DATA_DIR

MODEL_DIR = os.path.join(CURR_DIR, "results", "models_singlelayer_v1")
os.makedirs(MODEL_DIR, exist_ok=True)

torch.set_default_dtype(torch.float64)

def set_seeds(rs: int):
    torch.manual_seed(rs)
    np.random.seed(rs)

@dataclass(frozen=True)
class NSDEHParams:
    n_reparam_samples: int
    drift_layer_description: list[int]
    nonlinearity: nn.Module
    tau: float
    n_quad: int
    quad_percent: float
    n_tau: int

# @dataclass(frozen=True)
# class TrainHParams:
#     num_iters: int
#     transition_iters: int
#     batch_size: int
#     summary_freq: int
#     lr: float


#gruv1hppara
# @dataclass(frozen=True)
# class TrainHParams:
#     num_iters: int = 15000       # 增加总训练迭代次数
#     transition_iters: int = 5000  # 延长KL散度过渡期
#     batch_size: int = 32    # 减小批次大小适应序列特性
#     summary_freq: int = 200       # 调整日志频率
#     lr: float = 3e-4             # 降低初始学习率
#     grad_clip: float = 1.5      # 新增梯度裁剪阈值
#     weight_decay: float = 5e-5    # 新增权重衰减
#     patience_limit: int = 10


#lstmv1hppara
# @dataclass(frozen=True)
# class TrainHParams:
#     num_iters: int = 40000        # 增加总迭代次数
#     transition_iters: int = 10000 # 延长KL散度过渡期
#     batch_size: int = 48          # 折中的批次大小
#     summary_freq: int = 500
#     lr: float = 2e-4             # 降低学习率
#     grad_clip: float = 2.0        # 温和的梯度裁剪
#     weight_decay: float = 1e-4    # 保持权重衰减
#     patience_limit: int = 8       # 延长早停耐心值

# def get_hparams() -> tuple[NSDEHParams, TrainHParams]:
#     sde_hparams = NSDEHParams(
#     n_reparam_samples = 32,
#     drift_layer_description = [128, ],
#     nonlinearity = nn.Tanh(),
#     tau = 1e-5,
#     n_quad = 200,
#     quad_percent = 0.5,
#     n_tau = 500)
#     train_hparams = TrainHParams()
#     return sde_hparams, train_hparams


#单层嵌套版本
@dataclass(frozen=True)
class TrainHParams:
    num_iters: int
    batch_size: int
    summary_freq: int
    lr: float
    hidden_dim: int  # 新增：网络隐藏层维度

def get_hparams() -> tuple[NSDEHParams, TrainHParams]:
    sde_hparams = NSDEHParams(
            n_reparam_samples = 32,
            drift_layer_description = [128, ],
            nonlinearity = nn.Tanh(),
            tau = 1e-5,
            n_quad = 200,
            quad_percent = 0.5,
            n_tau = 500)
    train_hparams = TrainHParams(
        num_iters=2000,
        batch_size=64,
        summary_freq=100,
        lr=1e-3,
        hidden_dim=128  # 新增
    )
    return sde_hparams, train_hparams

#原始指定版本
# def get_hparams() -> tuple[NSDEHParams, TrainHParams]:
#     sde_hparams = NSDEHParams(
#     n_reparam_samples = 32,
#     drift_layer_description = [128, ],
#     nonlinearity = nn.Tanh(),
#     tau = 1e-5,
#     n_quad = 200,
#     quad_percent = 0.5,
#     n_tau = 500)
#     train_hparams = TrainHParams(
#     num_iters = int(20000),
#     transition_iters = 5000,
#     batch_size = 64,
#     summary_freq = 100,
#     lr = 1e-3)
#     return sde_hparams, train_hparams




#日志文件
def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    path_to_file = os.path.join(MODEL_DIR, "train.log")
    # Create a file handler
    fh = logging.FileHandler(path_to_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_data() -> dict[str, Any]:
    with open(os.path.join(DATA_DIR, "encoded_data.pkl"), "rb") as handle:
        data = pkl.load(handle)
    dtype = torch.float64
    return dict(num_data = data["z"].shape[0],
                d = data["z"].shape[1],
                t = data["t"].to(dtype),
                y_data = data["z"].to(dtype),
                var = data["code_stdev"].pow(2).to(dtype),
                t_span = (float(data["t"].min()), float(data["t"].max())))

def get_init_state(data: dict[str, Any], sde_hparams: NSDEHParams) -> dict[str, Any]:
    data_init_state = dict(d=data["d"],
                           t_span=data["t_span"], 
                           G=torch.eye(data["d"]),
                           measurement_noise=data["var"], 
                           train_t=data["t"], 
                           train_x=data["y_data"])
    return {**data_init_state, **asdict(sde_hparams)}

#原始模型初始化
# def initialize_model(init_state: dict[str, Any]) -> NeuralSDE:
#     nsde = NeuralSDE(**init_state)
#     nsde.train()
#     return nsde

def initialize_model(init_state: dict[str, Any]) -> SingleLayerSDELearner:
    """初始化使用单层SVISE先验的模型"""
    nsde = SingleLayerSDELearner(
        d=init_state["d"],
        t_span=init_state["t_span"],
        hidden_dim=128,  # 可调整
        n_reparam_samples=init_state["n_reparam_samples"],
        tau=init_state["tau"],
        n_quad=init_state["n_quad"],
        quad_percent=init_state["quad_percent"]
    )
    return nsde

def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

@no_grad
def plot_summary(curr_iter: int, device: torch.device, nsde: NeuralSDE, data: dict[str, Any]):
    plot_dir = os.path.join(MODEL_DIR, "plots", f"summary_{curr_iter}")
    os.makedirs(plot_dir, exist_ok=True)
    nsamples = nsde.n_reparam_samples
    ys_train = nsde.marginal_sde.generate_samples(data["t"].to(device), nsamples)

    median = torch.quantile(ys_train, 0.5, dim=0)
    n_plots = median.shape[1]
    max_plots_per_fig = 10
    n_figs = n_plots // max_plots_per_fig + (n_plots % max_plots_per_fig > 0)

    for i in range(n_figs):
        n_plots_in_this_fig = min(max_plots_per_fig, n_plots-i*max_plots_per_fig)
        fig, axes = plt.subplots(n_plots_in_this_fig, 1, figsize=(24, 3))
        for j in range(n_plots_in_this_fig):
            mode_idx = i*max_plots_per_fig+j
            if mode_idx < n_plots:  # Check if index is within range
                axes[j].plot(data["t"], data["y_data"][:, mode_idx])
                axes[j].plot(data["t"], median[:, mode_idx].cpu())
        plot_name = os.path.join(plot_dir, f"pod_median_{i}.png")
        fig.savefig(plot_name)
        plt.close(fig)

def cuda(device, iterable):
    for x in iterable:
        yield (xi.to(device) for xi in x)

def save_checkpoint(curr_iter: int, nsde: NeuralSDE):
    ckpt_path = os.path.join(MODEL_DIR, f"nsde_{curr_iter:06d}.pt")
    torch.save(nsde.state_dict(), ckpt_path)

def train_step(nsde: NeuralSDE, optimizer: torch.optim.Optimizer, batch: tuple[Tensor, Tensor], beta: float, num_data:int):
    t_batch, y_batch = batch
    optimizer.zero_grad()
    loss = -nsde.elbo(t_batch, y_batch, beta, num_data, compat_mode=False)
    loss.backward()
    optimizer.step()
    return loss

#单层嵌套训练
def train(nsde: SingleLayerSDELearner, hparams: TrainHParams, data: dict[str, Any], device: torch.device):
    nsde.to(device)
    logger = logging.getLogger(__name__)
    summary_freq = 1000
    num_iters = hparams.num_iters
    transition_iters = num_iters // 4  # 假设过渡迭代次数为总迭代次数的四分之一
    scheduler_freq = transition_iters // 2
    optimizer = torch.optim.Adam(
        [
            {"params": nsde.state_params()},
            {"params": nsde.sde_params(), "lr": hparams.lr},
        ],
        lr=hparams.lr / 10,
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = TensorDataset(data["t"], data["y_data"])
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    num_epochs = num_iters // len(train_loader)
    j = 0
    for _ in range(num_epochs):
        for t_batch, y_batch in train_loader:
            t_batch, y_batch = t_batch.to(device), y_batch.to(device)
            j += 1
            beta = min(1.0, (1.0 * j) / (transition_iters))
            if j % scheduler_freq == 0:
                scheduler.step()
            loss = nsde.loss(t_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % summary_freq == 0:
                nsde.eval()
                log_msg = f"Iter {j:06d} | beta: {beta:.2e} | loss {loss.item():.2e}"
                logger.info(log_msg)
                save_checkpoint(j, nsde)
                nsde.train()
    save_checkpoint(j, nsde)

#gruv1train设置（单一时序神经网络训练）
# def train(nsde: NeuralSDE, hparams: TrainHParams, data: dict[str, Any], device: torch.device):
#     # 初始化设备和模型
#     nsde.to(device)
#     logger = logging.getLogger(__name__)
#
#     # 验证训练参数
#     assert hparams.transition_iters < hparams.num_iters, "Transition iterations should be less than total iterations"
#
#     #gru
#     # 初始化优化器和学习率调度器
#     optimizer = torch.optim.AdamW(
#         [
#             {"params": nsde.state_params()},
#             {"params": nsde.sde_params(), "lr": hparams.lr},
#         ],
#         lr=hparams.lr / 10,
#         weight_decay=hparams.weight_decay
#     )
#
#     scheduler = lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.5,
#         patience=5,
#     )
#
#     #lstm
#     # 使用AdamW优化器（更适合LSTM）
#     #
#     # optimizer = torch.optim.AdamW(
#     #     [
#     #         {"params": nsde.state_params()},
#     #         {"params": nsde.sde_params(), "lr": hparams.lr},
#     #     ],
#     #     lr=hparams.lr / 10,
#     #     weight_decay=hparams.weight_decay
#     # )
#     #
#     # # 使用余弦退火调度器
#     # scheduler = lr_scheduler.CosineAnnealingLR(
#     #     optimizer,
#     #     T_max=hparams.num_iters,
#     #     eta_min=1e-5
#     # )
#
#     # 准备数据加载器
#     train_dataset = TensorDataset(data["t"], data["y_data"])
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=hparams.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=0
#     )
#
#     # 训练参数
#     num_epochs = hparams.num_iters // len(train_loader) + 1
#     best_loss = float('inf')
#     patience_counter = 0
#     patience_limit = hparams.patience_limit  # 早停耐心值
#
#     # 训练循环
#     nsde.train()
#     global_step = 0
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for t_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
#             global_step += 1
#             # 转移到GPU
#             t_batch = t_batch.to(device)
#             y_batch = y_batch.to(device)
#             # 计算动态beta值
#             beta = min(1.0, (1.0 * global_step) / hparams.transition_iters)
#             # 训练步骤
#             optimizer.zero_grad()
#             loss = -nsde.elbo(t_batch, y_batch, beta, data["num_data"])
#             # 反向传播
#             loss.backward()
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(nsde.parameters(), hparams.grad_clip)
#             # 参数更新
#             optimizer.step()
#             # 学习率调整
#             # scheduler.step()
#             # 记录损失
#             epoch_loss += loss.item()
#             # 日志和保存
#             if global_step % hparams.summary_freq == 0:
#                 # 验证步骤
#                 nsde.eval()
#                 with torch.no_grad():
#                     val_loss = -nsde.elbo(data["t"].to(device), data["y_data"].to(device),
#                                           beta=1.0,N=data["num_data"])
#                 # 保存最佳模型
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     save_checkpoint(global_step, nsde)
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#
#                 #re'ducelr
#                 scheduler.step(val_loss.item())
#
#                 # 早停检查
#                 if patience_counter >= patience_limit:
#                     logger.info(f"Early stopping at step {global_step}")
#                     return
#                 # 记录日志
#                 logger.info(
#                     f"Iter {global_step:06d} | "
#                     f"Train Loss: {loss.item():.3e} | "
#                     f"Val Loss: {val_loss.item():.3e} | "
#                     f"LR: {optimizer.param_groups[0]['lr']:.1e}"
#                 )
#
#                 # 生成可视化
#                 if global_step % (hparams.summary_freq * 5) == 0:
#                     plot_summary(global_step, device, nsde, data)
#                 nsde.train()
#         # 周期统计
#         epoch_loss /= len(train_loader)
#         logger.info(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.3e}")
#     # 最终保存
#     save_checkpoint(global_step, nsde)
#     logger.info("Training completed.")


#原始fcnn训练过程
# def train(nsde: NeuralSDE, hparams: TrainHParams, data: dict[str, Any], device: torch.device):
#     nsde.to(device)
#     assert hparams.transition_iters < hparams.num_iters
#     logger = logging.getLogger(__name__)
#     summary_freq = 1000
#     scheduler_freq = hparams.transition_iters // 2
#     optimizer = torch.optim.Adam(
#         [
#             {"params": nsde.state_params()},
#             {"params": nsde.sde_params(), "lr": hparams.lr},
#         ],
#         lr=hparams.lr / 10,
#     )
#     scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#     train_dataset = TensorDataset(data["t"], data["y_data"])
#     train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
#     num_epochs = hparams.num_iters // len(train_loader)
#     j = 0
#     for _ in range(num_epochs):
#         for t_batch, y_batch in cuda(device, train_loader):
#             j += 1
#             beta = min(1.0, (1.0 * j) / (hparams.transition_iters))
#             if j % scheduler_freq == 0:
#                 scheduler.step()
#             loss = train_step(nsde, optimizer, (t_batch, y_batch), beta, data["num_data"])
#             if j % summary_freq == 0:
#                 nsde.eval()
#                 log_msg = f"Iter {j:06d} | beta: {beta:.2e} | loss {loss.item():.2e}"
#                 logger.info(log_msg)
#                 save_checkpoint(j, nsde)
#                 nsde.train()
#     save_checkpoint(j, nsde)
            

def main():
    set_seeds(23)
    logger = setup_logger()
    data = get_data()
    sde_hparams, train_hparams = get_hparams()
    init_state = get_init_state(data, sde_hparams)
    with open(os.path.join(MODEL_DIR,"init_state_freeze.pkl"), "wb") as handle:
        pkl.dump(init_state, handle)

    # nsde = initialize_model(init_state)
    # 新的模型类
    nsde = initialize_model({
        "d": data["y_data"].shape[1],
        "t_span": (data["t"][0].item(), data["t"][-1].item()),
        "n_reparam_samples": sde_hparams.n_reparam_samples,
        "tau": sde_hparams.tau,
        "n_quad": sde_hparams.n_quad,
        "quad_percent": sde_hparams.quad_percent
    })

    train(nsde, train_hparams, data, torch.device("cuda:0"))
    logger.info("Done.")
    print("Done.")


if __name__ == "__main__":
    main()