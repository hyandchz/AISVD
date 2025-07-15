import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """修正的复数卷积层，处理实部和虚部作为两个通道"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        """
        参数:
          in_channels: 输入通道数（对于复数数据应为2）
          out_channels: 输出通道数（每个通道对应实部或虚部）
          kernel_size: 卷积核大小
          padding: 填充大小
        """
        super().__init__()
        # 创建处理实部和虚部的卷积层
        # 输入通道应为2（实部和虚部）
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        """
        输入: [batch, in_channels, H, W] (in_channels应为2)
        输出: [batch, out_channels*2, H, W] (拼接后的实部和虚部)
        """
        # 分别计算实部和虚部
        real_part = self.conv_real(x) - self.conv_imag(x)
        imag_part = self.conv_real(x) + self.conv_imag(x)

        # 拼接结果
        return torch.cat([real_part, imag_part], dim=1)


class GramSchmidtLayer(nn.Module):
    """Gram-Schmidt正交化层，确保奇异矩阵满足正交约束"""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, matrix):
        batch_size, M, R, _ = matrix.shape
        orthogonal_matrix = torch.zeros_like(matrix)

        for b in range(batch_size):
            vectors = []
            for j in range(R):
                v_real = matrix[b, :, j, 0].clone()
                v_imag = matrix[b, :, j, 1].clone()

                for k in range(len(vectors)):
                    basis_real, basis_imag = vectors[k]

                    # 复数点积: v·basis[k]^H
                    dot_real = torch.dot(v_real, basis_real) + torch.dot(v_imag, basis_imag)
                    dot_imag = torch.dot(v_imag, basis_real) - torch.dot(v_real, basis_imag)

                    # 投影系数
                    proj_coef = dot_real / (basis_real.norm() ** 2 + basis_imag.norm() ** 2 + self.epsilon)

                    # 减去投影分量
                    v_real = v_real - proj_coef * basis_real
                    v_imag = v_imag - proj_coef * basis_imag

                # 归一化
                norm = torch.sqrt(torch.sum(v_real ** 2 + v_imag ** 2) + self.epsilon)
                v_real = v_real / norm
                v_imag = v_imag / norm

                vectors.append((v_real, v_imag))

            # 构造正交矩阵
            for j in range(R):
                orthogonal_matrix[b, :, j, 0] = vectors[j][0]
                orthogonal_matrix[b, :, j, 1] = vectors[j][1]

        return orthogonal_matrix


class SVDNet(nn.Module):
    """
    无线鲁棒SVD神经网络模型
    输入: 非理想信道矩阵 [batch, M, N, 2] (实部和虚部)
    输出: U[batch, M, R, 2], S[batch, R], V[batch, N, R, 2]
    """

    def __init__(self, M, N, R, feature_dim=128):
        super(SVDNet, self).__init__()
        self.M = M
        self.N = N
        self.R = R

        # 修正的复数卷积层 - 输入通道设为2（实部和虚部）
        self.conv_block = nn.Sequential(
            # 第一层：输入通道=2（实部和虚部），输出通道=32（实际输出64通道：32实+32虚）
            ComplexConv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 输入通道为64（32实+32虚）
            nn.ReLU(inplace=True),

            # 第二层：输入通道=64，输出通道=64（实际输出128通道）
            ComplexConv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 第三层：输入通道=128，输出特征维度
            ComplexConv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True)
        )

        # U分支输出 (左奇异矩阵)
        self.u_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((M, 1)),  # 输出形状: [batch, 2*feature_dim, M, 1]
            nn.Conv2d(feature_dim * 2, 2 * R, kernel_size=1),  # 输出形状: [batch, 2*R, M, 1]
            nn.ReLU(inplace=True)
        )

        # S分支输出 (奇异值)
        self.s_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出形状: [batch, 2*feature_dim, 1, 1]
            nn.Flatten(),  # 输出形状: [batch, 2*feature_dim]
            nn.Linear(feature_dim * 2, R),  # 输出形状: [batch, R]
            nn.ReLU(inplace=True)
        )

        # V分支输出 (右奇异矩阵)
        self.v_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, N)),  # 输出形状: [batch, 2*feature_dim, 1, N]
            nn.Conv2d(feature_dim * 2, 2 * R, kernel_size=1),  # 输出形状: [batch, 2*R, 1, N]
            nn.ReLU(inplace=True)
        )

        # 正交约束层
        self.orthogonal_layer = GramSchmidtLayer()

    def forward(self, x):
        # 输入形状转换: [batch, M, N, 2] -> [batch, 2, M, N]
        x = x.permute(0, 3, 1, 2)

        # 特征提取 - 输入应为 [batch, 2, M, N]
        x = self.conv_block(x)

        # U分支处理
        u = self.u_branch(x)  # [batch, 2*R, M, 1]
        u = u.permute(0, 2, 3, 1)  # [batch, M, 1, 2*R]
        u = u.view(-1, self.M, self.R, 2)  # [batch, M, R, 2]
        u = self.orthogonal_layer(u)  # 应用正交约束

        # S分支处理
        s = self.s_branch(x)  # [batch, R]

        # V分支处理
        v = self.v_branch(x)  # [batch, 2*R, 1, N]
        v = v.permute(0, 3, 2, 1)  # [batch, N, 1, 2*R]
        v = v.view(-1, self.N, self.R, 2)  # [batch, N, R, 2]
        v = self.orthogonal_layer(v)  # 应用正交约束

        return u, s, v

    @staticmethod
    def complex_matrix_multiplication(u, s, v):
        """复数矩阵乘法: 重构信道矩阵 H = U * diag(S) * V^H"""
        # 计算 V 的共轭转置 V^H
        v_conj = v.clone()
        v_conj[..., 1] = -v_conj[..., 1]  # 虚部取负得到共轭
        v_h = v_conj.permute(0, 2, 1, 3)  # [batch, R, N, 2] 转置

        # 计算 diag(S) * V^H
        s_expanded = s.unsqueeze(-1).unsqueeze(-1)  # [batch, R, 1, 1]
        diag_s_vh = s_expanded * v_h  # [batch, R, N, 2]

        # 拆分实虚部
        diag_s_vh_real = diag_s_vh[..., 0]  # [batch, R, N]
        diag_s_vh_imag = diag_s_vh[..., 1]  # [batch, R, N]

        # 拆分U的实虚部
        u_real = u[..., 0]  # [batch, M, R]
        u_imag = u[..., 1]  # [batch, M, R]

        # 计算 U * (diag(S) * V^H)
        h_real = torch.matmul(u_real, diag_s_vh_real) - torch.matmul(u_imag, diag_s_vh_imag)
        h_imag = torch.matmul(u_real, diag_s_vh_imag) + torch.matmul(u_imag, diag_s_vh_real)

        return h_real, h_imag

    @staticmethod
    def orthogonal_constraint_loss(u, v):
        """计算正交性约束损失: ||U^H U - I||_F^2 + ||V^H V - I||_F^2"""

        def orth_loss(matrix):
            real = matrix[..., 0]  # [batch, dim1, dim2]
            imag = matrix[..., 1]  # [batch, dim1, dim2]

            # 计算 matrix^H * matrix
            prod_real = torch.matmul(real.transpose(1, 2), real) + torch.matmul(imag.transpose(1, 2), imag)
            prod_imag = torch.matmul(real.transpose(1, 2), imag) - torch.matmul(imag.transpose(1, 2), real)

            # 目标: 单位矩阵
            identity = torch.eye(matrix.size(2), device=matrix.device).unsqueeze(0)
            zero_matrix = torch.zeros_like(prod_imag)

            # 计算Frobenius范数平方
            loss_real = torch.norm(prod_real - identity, p='fro') ** 2
            loss_imag = torch.norm(prod_imag - zero_matrix, p='fro') ** 2

            return loss_real + loss_imag

        u_loss = orth_loss(u)  # U的正交损失
        v_loss = orth_loss(v)  # V的正交损失
        return u_loss + v_loss


def train_model(model, train_data, train_label, device, epochs=100, batch_size=32):
    """训练模型函数"""
    from torch.utils.data import TensorDataset, DataLoader

    # 确保数据形状正确
    assert train_data.shape[-1] == 2, "输入数据最后一维应为实虚部(2)"
    assert train_label.shape[-1] == 2, "标签数据最后一维应为实虚部(2)"

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 准备数据加载器
    dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_label, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_ortho_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            u, s, v = model(inputs)

            # 重构信道矩阵
            h_real, h_imag = SVDNet.complex_matrix_multiplication(u, s, v)
            h_recon = torch.stack([h_real, h_imag], dim=-1)

            # 计算损失
            recon_loss = F.mse_loss(h_recon, labels)
            ortho_loss = SVDNet.orthogonal_constraint_loss(u, v)
            loss = recon_loss + 0.1 * ortho_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_ortho_loss += ortho_loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_ortho = total_ortho_loss / len(dataloader)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Loss: {avg_loss:.6f}, '
              f'Recon: {avg_recon:.6f}, '
              f'Ortho: {avg_ortho:.6f}')

    return model