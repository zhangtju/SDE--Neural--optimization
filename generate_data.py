from __future__ import annotations
from dataclasses import dataclass
import os
import pathlib
import pickle as pkl

import torch
from torch import Tensor
import numpy as np

from svise import pca

CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
DATA_PATH = os.path.join(CURR_DIR, "data", "vortex.pkl")

DATA_DIR = os.path.join(CURR_DIR, "data")


@dataclass(frozen=True)
class FlowDataContainer:
    """Container for flow data"""

    t: Tensor
    x: Tensor
    y: Tensor
    vel_x: Tensor
    vel_y: Tensor
    train_y: Tensor
    vorticity: Tensor


@dataclass(frozen=True)
class CylinderFlowData:
    """storage container for raw cylinder flow data"""
    train_data: FlowDataContainer
    test_data: FlowDataContainer
    grid_shape: tuple[int, int] = (199, 1499)
    adjusted_grid: tuple[int ,int] = (199, 1499)

    @classmethod
    def load_from_file(
        cls, file_path: str, skip_percent: float = 0.2, test_percent: float = 0.2, dtype: torch.dtype = torch.float64
    ) -> CylinderFlowData:
        """loads a cylinder dataset from a file_path"""
        with open(file_path, "rb") as handle:
            data = pkl.load(handle)

        def filter_by_name(name):
            flattened_data = torch.tensor(data["x"][..., data["var_names"].index(name)], dtype=dtype)
            batch_size = flattened_data.shape[0]
            flattened_data = flattened_data.reshape(batch_size, *cls.grid_shape)[:, :, :cls.adjusted_grid[1]]
            return flattened_data.reshape(batch_size, -1)

        assert 0 <= skip_percent < 1, "skip must be in range [0, 1)"

        def remove_percent(data):
            # the first couple of frames have some transients that
            # are highly nonlinear, let's just ignore them for now
            skip = int(len(data) * skip_percent)
            return data[skip:]

        def get_train(data):
            data = remove_percent(data)
            test = int(len(data) * test_percent)
            return data[:-test]

        def get_test(data):
            data = remove_percent(data)
            test = int(len(data) * test_percent)
            return data[-test:]

        def stack_velocity():
            u = filter_by_name("u")
            v = filter_by_name("v")
            return torch.cat([u, v], dim=1)

        t = torch.tensor([0.1 * j for j in range(data["x"].shape[0])])
        return cls(
            train_data=FlowDataContainer(
                t=get_train(t),
                x=get_train(filter_by_name("x")),
                y=get_train(filter_by_name("y")),
                vel_x=get_train(filter_by_name("u")),
                vel_y=get_train(filter_by_name("v")),
                train_y=get_train(stack_velocity()),
                vorticity=get_train(filter_by_name("Vorticity")),
            ),
            test_data=FlowDataContainer(
                t=get_test(t),
                x=get_test(filter_by_name("x")),
                y=get_test(filter_by_name("y")),
                vel_x=get_test(filter_by_name("u")),
                vel_y=get_test(filter_by_name("v")),
                train_y=get_test(stack_velocity()),
                vorticity=get_test(filter_by_name("Vorticity")),
            ),
        )


def main():
    rs = 21
    torch.manual_seed(rs)
    np.random.seed(rs)
    print("Loading data...")
    data = CylinderFlowData.load_from_file(DATA_PATH, dtype=torch.float32)
    # ---------------------- PCA ----------------------
    train_y = data.train_data.train_y
    print("Performing PCA...")
    lin_model, z = pca.PCA.create(train_y, percent_cutoff=0.9, max_evecs=100)
    # assume 0.001 % error in code vectors.
    code_stdev = torch.ones_like(z[0]) * 1e-3
    valid_y = data.test_data.train_y
    x_grid = data.train_data.x[0].reshape(*data.adjusted_grid)[0]
    y_grid = data.train_data.y[0].reshape(*data.adjusted_grid)[:, 0]
    data = {
        "z": z,
        "code_stdev": code_stdev,
        "lin_model": lin_model.state_dict(),
        "t": data.train_data.t,
        "valid_t": data.test_data.t,
        "valid_z": lin_model.encode(valid_y),
        "grid": (x_grid, y_grid),
        "valid_y": data.test_data.train_y,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "encoded_data.pkl"), "wb") as f:
        pkl.dump(data, f)
    print(f"Done, found {data['z'].shape[1]} modes.")

if __name__ == "__main__":
    main()


#new for lower cpu
# from __future__ import annotations
# from dataclasses import dataclass
# import os
# import pathlib
# import pickle as pkl
# import gc
#
# import torch
# from torch import Tensor
# import numpy as np
#
# from svise import pca
#
# CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
# DATA_PATH = os.path.join(CURR_DIR, "data", "vortex.pkl")
# DATA_DIR = os.path.join(CURR_DIR, "data")
#
# # 配置参数
# DTYPE = torch.float32  # 保持单精度
# CHUNK_SIZE = 500       # 减小分块大小
#
#
# @dataclass(frozen=True)
# class FlowDataContainer:
#     """Container for flow data"""
#     t: Tensor
#     x: Tensor
#     y: Tensor
#     vel_x: Tensor
#     vel_y: Tensor
#     train_y: Tensor
#     vorticity: Tensor
#
#
# @dataclass(frozen=True)
# class CylinderFlowData:
#     """storage container for raw cylinder flow data"""
#     train_data: FlowDataContainer
#     test_data: FlowDataContainer
#     grid_shape: tuple[int, int] = (199, 1499)
#     adjusted_grid: tuple[int, int] = (199, 1499)
#
#     @classmethod
#     def load_from_file(
#             cls, file_path: str, skip_percent: float = 0.2,
#             test_percent: float = 0.2, dtype: torch.dtype = DTYPE
#     ) -> CylinderFlowData:
#         with open(file_path, "rb") as handle:
#             data = pkl.load(handle)
#
#         # 核心修改：分阶段处理避免全量合并
#         def process_variable(name, apply_skip=True):
#             """分块处理单个变量，直接返回训练和测试数据"""
#             total_samples = data["x"].shape[0]
#             var_idx = data["var_names"].index(name)
#
#             # 第一阶段：计算数据划分位置
#             skip = int(total_samples * skip_percent) if apply_skip else 0
#             test_split = int((total_samples - skip) * (1 - test_percent))
#
#             # 第二阶段：分块处理训练和测试数据
#             train_chunks, test_chunks = [], []
#             for i in range(0, total_samples, CHUNK_SIZE):
#                 chunk_slice = slice(i, min(i + CHUNK_SIZE, total_samples))
#
#                 # 原始数据分块
#                 arr = data["x"][chunk_slice, ..., var_idx]
#                 tensor = torch.tensor(arr, dtype=dtype)
#                 tensor = tensor.reshape(-1, *cls.grid_shape)[:, :, :cls.adjusted_grid[1]]
#                 tensor = tensor.reshape(-1, tensor.shape[-1])
#
#                 # 直接划分到训练/测试集
#                 if apply_skip:
#                     chunk_skip = max(0, skip - i)
#                     chunk_test_split = max(0, test_split - i)
#
#                     train_part = tensor[chunk_skip:chunk_test_split]
#                     test_part = tensor[chunk_test_split:]
#                 else:
#                     train_part = tensor
#                     test_part = tensor
#
#                 if len(train_part) > 0:
#                     train_chunks.append(train_part)
#                 if len(test_part) > 0:
#                     test_chunks.append(test_part)
#
#                 del tensor, arr
#                 gc.collect()
#
#             # 返回分块数据避免合并
#             return train_chunks, test_chunks
#
#         # 处理时间序列
#         t = torch.linspace(0, 0.1 * (data["x"].shape[0] - 1), data["x"].shape[0], dtype=dtype)
#         skip = int(len(t) * skip_percent)
#         test_split = len(t) - int((len(t) - skip) * test_percent)
#         train_t, test_t = t[skip:test_split], t[test_split:]
#
#         # 处理所有变量（保持分块状态）
#         variables = ["x", "y", "u", "v", "Vorticity"]
#         processed = {var: process_variable(var) for var in variables}
#
#         # 处理速度场（特殊处理）
#         def process_velocity():
#             u_train, u_test = process_variable("u", apply_skip=False)
#             v_train, v_test = process_variable("v", apply_skip=False)
#
#             # 分块合并速度场
#             velocity_train = [torch.cat([u, v], dim=1) for u, v in zip(u_train, v_train)]
#             velocity_test = [torch.cat([u, v], dim=1) for u, v in zip(u_test, v_test)]
#
#             # 应用数据划分
#             skip_vel = int(len(velocity_train) * skip_percent / CHUNK_SIZE)
#             test_split_vel = len(velocity_train) - int((len(velocity_train) - skip_vel) * test_percent)
#             return velocity_train[skip_vel:test_split_vel], velocity_test[test_split_vel:]
#
#         velocity_train, velocity_test = process_velocity()
#
#         # 构建数据容器（保持分块状态）
#         return cls(
#             train_data=FlowDataContainer(
#                 t=train_t,
#                 x=torch.cat(processed["x"][0], dim=0),
#                 y=torch.cat(processed["y"][0], dim=0),
#                 vel_x=torch.cat(processed["u"][0], dim=0),
#                 vel_y=torch.cat(processed["v"][0], dim=0),
#                 train_y=torch.cat(velocity_train, dim=0),
#                 vorticity=torch.cat(processed["Vorticity"][0], dim=0),
#             ),
#             test_data=FlowDataContainer(
#                 t=test_t,
#                 x=torch.cat(processed["x"][1], dim=0),
#                 y=torch.cat(processed["y"][1], dim=0),
#                 vel_x=torch.cat(processed["u"][1], dim=0),
#                 vel_y=torch.cat(processed["v"][1], dim=0),
#                 train_y=torch.cat(velocity_test, dim=0),
#                 vorticity=torch.cat(processed["Vorticity"][1], dim=0),
#             ),
#         )
#
#
# def main():
#     rs = 21
#     torch.manual_seed(rs)
#     np.random.seed(rs)
#     print("Loading data...")
#     data = CylinderFlowData.load_from_file(DATA_PATH, dtype=torch.float32)
#
#     # 检查点1：验证数据加载是否成功
#     print("[DEBUG] 数据加载完成，检查关键字段...")
#     print(f"训练数据形状: {data.train_data.train_y.shape if data.train_data else '空'}")
#     print(f"测试数据形状: {data.test_data.train_y.shape if data.test_data else '空'}")
#
#     # 检查点2：验证PCA输入
#     train_y = data.train_data.train_y
#     if train_y.nelement() == 0:
#         print("[ERROR] 训练数据为空!")
#         return
#
#     # ---------------------- PCA ----------------------
#     print("Performing PCA...")
#     try:
#         lin_model, z = pca.PCA.create(train_y, percent_cutoff=0.9, max_evecs=100)
#         print(f"[DEBUG] PCA完成，潜在变量形状: {z.shape}")
#     except Exception as e:
#         print(f"[ERROR] PCA失败: {str(e)}")
#         return
#     lin_model, z = pca.PCA.create(train_y, percent_cutoff=0.9, max_evecs=100)
#     # assume 0.001 % error in code vectors.
#     code_stdev = torch.ones_like(z[0]) * 1e-3
#     valid_y = data.test_data.train_y
#     x_grid = data.train_data.x[0].reshape(*data.adjusted_grid)[0]
#     y_grid = data.train_data.y[0].reshape(*data.adjusted_grid)[:, 0]
#     data = {
#         "z": z.to(DTYPE),
#         "code_stdev": code_stdev.to(DTYPE),
#         "lin_model": lin_model.state_dict(),
#         "t": data.train_data.t.to(DTYPE),
#         "valid_t": data.test_data.t.to(DTYPE),
#         "valid_z": lin_model.encode(valid_y).to(DTYPE),
#         "grid": (x_grid.to(DTYPE), y_grid.to(DTYPE)),
#         "valid_y": data.test_data.train_y.to(DTYPE),
#     }
#
#
# if __name__ == "__main__":
#     main()