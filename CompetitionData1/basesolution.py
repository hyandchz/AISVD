import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

batch_size = 128

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDNet(nn.Module):
    def __init__(self):
        """
        符合比赛平台要求的SVD网络实现
        - 无参初始化：支持 model = SVDNet()
        - 固定输入输出尺寸：64x64 MIMO矩阵，输出32个奇异值
        """
        super(SVDNet, self).__init__()
        # 固定参数设置
        self.M = 64  # 固定行数
        self.N = 64  # 固定列数
        self.R = 32  # 固定奇异值数量
        feature_dim = 16

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.Conv2d(8, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # U分支：输出左奇异矩阵
        self.u_branch = nn.Conv2d(feature_dim, 2 * self.R, kernel_size=1)

        # S分支：输出奇异值向量
        self.s_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, self.R)
        )

        # V分支：输出右奇异矩阵
        self.v_branch = nn.Conv2d(feature_dim, 2 * self.R, kernel_size=1)

    def forward(self, x):
        """
        前向传播 (兼容单样本和批量输入)

        输入:
          - 单样本: [M, N, 2] = [64, 64, 2]
          - 批量样本: [batch, M, N, 2] = [batch, 64, 64, 2]

        输出:
          u: [64, 32, 2] 或 [batch, 64, 32, 2]
          s: [32] 或 [batch, 32]
          v: [64, 32, 2] 或 [batch, 64, 32, 2]
        """
        # 处理单样本输入
        if x.dim() == 3:
            batch_size = 1
            x = x.unsqueeze(0)  # 增加batch维度 [1, 64, 64, 2]
        else:
            batch_size = x.size(0)

        # 输入转换: [batch, 64, 64, 2] -> [batch, 2, 64, 64]
        x = x.permute(0, 3, 1, 2)

        # 特征提取
        features = self.feature_extractor(x)

        # U分支处理
        u = self.u_branch(features)
        u = u.permute(0, 2, 3, 1)
        u = u.reshape(batch_size, 64, 64, self.R, 2)
        u = u.mean(dim=2)  # [batch, 64, 32, 2]

        # S分支处理
        s = self.s_branch(features)  # [batch, 32]

        # V分支处理
        v = self.v_branch(features)
        v = v.permute(0, 2, 3, 1)
        v = v.reshape(batch_size, 64, 64, self.R, 2)
        v = v.mean(dim=2)  # [batch, 64, 32, 2]

        # 归一化处理
        u = self.normalize_columns(u)
        v = self.normalize_columns(v)

        # 如果是单样本，移除batch维度
        if batch_size == 1:
            u = u.squeeze(0)
            s = s.squeeze(0)
            v = v.squeeze(0)

        return u, s, v

    def normalize_columns(self, mat):
        """
        列向量归一化
        """
        real = mat[..., 0]
        imag = mat[..., 1]
        norm = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        real_norm = real / norm
        imag_norm = imag / norm
        return torch.stack([real_norm, imag_norm], dim=-1)

def orthogonal_constraint(u, v):
    """
    轻量级正交约束损失函数

    参数:
        u: 左奇异矩阵 [batch, M, R, 2]
        v: 右奇异矩阵 [batch, N, R, 2]

    返回:
        正交约束损失值
    """

    def orth_loss(matrix):
        # matrix: [batch, dim1, dim2, 2]
        real = matrix[..., 0]  # [batch, dim1, dim2]
        imag = matrix[..., 1]

        # 计算 matrix^H * matrix
        prod_real = torch.matmul(real.transpose(1, 2), real) + torch.matmul(imag.transpose(1, 2), imag)

        # 目标: 单位矩阵
        identity = torch.eye(prod_real.size(1), device=matrix.device).unsqueeze(0)
        identity = identity.expand(batch_size, -1, -1)  # [batch_size, R, R]
        return F.mse_loss(prod_real, identity)

    return orth_loss(u) + orth_loss(v)


def complex_matrix_multiplication(u, s, v):
    """
    复数矩阵乘法重构信道矩阵: H = U * diag(S) * V^H
    """
    # 创建复数版本的 v_h
    v_real = v[..., 0].transpose(1, 2)  # [batch, R, N]
    v_imag = -v[..., 1].transpose(1, 2)  # 共轭：虚部取负
    v_h_complex = torch.view_as_complex(torch.stack((v_real, v_imag), dim=-1))

    # 创建复数版本的 diag_s
    s_diag = torch.diag_embed(s)  # [batch, R, R]
    s_diag_complex = torch.view_as_complex(
        torch.stack((s_diag, torch.zeros_like(s_diag)), dim=-1)
    )

    # 创建复数版本的 u
    u_real = u[..., 0]  # [batch, M, R]
    u_imag = u[..., 1]  # [batch, M, R]
    u_complex = torch.view_as_complex(torch.stack((u_real, u_imag), dim=-1))

    # 计算复数矩阵乘法
    diag_s_vh = torch.matmul(s_diag_complex, v_h_complex)  # [batch, R, N]
    h_complex = torch.matmul(u_complex, diag_s_vh)  # [batch, M, N]

    # 分离实部和虚部
    h_real = h_complex.real
    h_imag = h_complex.imag

    return h_real, h_imag


def train_model(model, train_data_list, train_label_list, device,
                           epochs_per_round=50, batch_size=128, patience=5,
                           model_save_path='final_model.pth'):
    """
    多轮数据训练函数（轮流训练多组数据）

    参数:
        model: 待训练模型
        train_data_list: 训练数据列表 [data1, data2, data3]
        train_label_list: 训练标签列表 [label1, label2, label3]
        device: 训练设备
        epochs_per_round: 每轮数据的训练轮数
        batch_size: 批大小
        patience: 早停机制等待轮数
        model_save_path: 最终模型保存路径
    """
    from torch.utils.data import TensorDataset, DataLoader

    # 训练多轮数据
    for round_idx, (train_data, train_label) in enumerate(zip(train_data_list, train_label_list)):
        print(f"\n{'=' * 50}")
        print(f"Starting Training for Round {round_idx + 1}/{len(train_data_list)}")
        print(f"Training samples: {train_data.shape[0]}")
        print(f"Channel matrix size: {train_data.shape[1]}x{train_data.shape[2]}")
        print(f"{'=' * 50}\n")

        # 准备数据
        dataset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_label, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_loss = float('inf')
        epochs_without_improvement = 0

        # 训练循环
        for epoch in range(epochs_per_round):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_ortho_loss = 0.0

            # 使用tqdm创建进度条
            batch_iterator = tqdm(dataloader, desc=f'Round {round_idx + 1} - Epoch {epoch + 1}/{epochs_per_round}',
                                  unit='batch', leave=True)

            for batch_idx, (inputs, labels) in enumerate(batch_iterator):
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                u, s, v = model(inputs)

                # 重构信道矩阵
                h_real, h_imag = complex_matrix_multiplication(u, s, v)
                h_recon = torch.stack([h_real, h_imag], dim=-1)

                # 计算损失
                recon_loss = F.mse_loss(h_recon, labels)
                ortho_loss = orthogonal_constraint(u, v)
                loss = recon_loss + 0.02 * ortho_loss  # 减小正交约束权重

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()

                # 更新损失统计
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_ortho_loss += ortho_loss.item()

                # 实时更新进度条描述
                avg_recon = total_recon_loss / (batch_idx + 1)
                avg_ortho = total_ortho_loss / (batch_idx + 1)
                avg_loss = total_loss / (batch_idx + 1)

                batch_iterator.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{avg_loss:.6f}',
                    'recon': f'{recon_loss.item():.6f}',
                    'ortho': f'{ortho_loss.item():.6f}'
                })

            # 计算本轮平均损失
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_ortho_loss = total_ortho_loss / len(dataloader)
            avg_loss = total_loss / len(dataloader)

            # 更新学习率
            scheduler.step(avg_loss)

            if scheduler.num_bad_epochs > 0 and scheduler.num_bad_epochs % scheduler.patience == 0:
                print(
                    f"  Learning rate reduced from {optimizer.param_groups[0]['lr'] / (scheduler.factor):.6f} to {optimizer.param_groups[0]['lr']:.6f}")

            # 打印本轮摘要
            print(f"\nRound {round_idx + 1} - Epoch {epoch + 1}/{epochs_per_round} Summary:")
            print(f"  Avg Loss: {avg_loss:.6f}")
            print(f"  Avg Recon Loss: {avg_recon_loss:.6f}")
            print(f"  Avg Ortho Loss: {avg_ortho_loss:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

            # 早停机制检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

                # 检查是否需要提前停止
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    break

        print(f"\n{'=' * 50}")
        print(f"Completed Training for Round {round_idx + 1}/{len(train_data_list)}")
        print(f"Final Loss: {avg_loss:.6f}")
        print(f"{'=' * 50}")

    # 保存最终模型
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")

    return model

