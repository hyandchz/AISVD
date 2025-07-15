import os
import numpy as np
import torch
from solution import SVDNet
from utils.data_loader import read_cfg_file, load_test_data, get_data_paths
from config import get_device, get_case_ids


def predict(model, test_data, device, batch_size=32):
    """批量预测函数"""
    model.eval()
    samp_num = test_data.shape[0]
    M = model.M
    N = model.N
    R = model.R

    U_out_all = np.zeros((samp_num, M, R, 2), dtype=np.float32)
    S_out_all = np.zeros((samp_num, R), dtype=np.float32)
    V_out_all = np.zeros((samp_num, N, R, 2), dtype=np.float32)

    # 批量处理数据以提高效率
    for i in range(0, samp_num, batch_size):
        end_idx = min(i + batch_size, samp_num)
        batch_data = test_data[i:end_idx]
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            U_out, S_out, V_out = model(batch_tensor)

        # 将结果存入数组
        batch_size_actual = end_idx - i
        U_out_all[i:end_idx] = U_out.cpu().numpy()
        S_out_all[i:end_idx] = S_out.cpu().numpy()
        V_out_all[i:end_idx] = V_out.cpu().numpy()

    return U_out_all, S_out_all, V_out_all


def main():
    print("<<< Starting Prediction Process >>>\n")

    # 获取数据路径
    PathRaw, Prefix = get_data_paths()
    device = get_device()
    print(f"Using device: {device}")

    # 获取所有场景ID
    case_ids = get_case_ids(PathRaw)

    # 处理每个场景
    for case_id in case_ids:
        print(f'\nProcessing Round {Ridx} Case {case_id}')

        # 读取配置文件
        cfg_path = os.path.join(PathRaw, f'{Prefix}CfgData{case_id}.txt')
        samp_num, M, N, IQ, R = read_cfg_file(cfg_path)

        # 加载测试数据
        test_data = load_test_data(PathRaw, Prefix, case_id)
        if test_data is None:
            print(f"No test data for Round {Ridx} Case {case_id}, skipping.")
            continue

        # 初始化模型
        model = SVDNet(M, N, R).to(device)
        model_path = f'model_round{Ridx}_case{case_id}.pth'

        # 加载模型权重（如果存在）
        if os.path.exists(model_path):
            print(f"Loading pre-trained model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Warning: Using untrained model for prediction")

        # 预测
        U_out, S_out, V_out = predict(model, test_data, device)

        # 保存结果
        TestOutput_file = f'{Prefix}TestOutput{case_id}.npz'
        np.savez(TestOutput_file, U_out=U_out, S_out=S_out, V_out=V_out)
        print(f"Results saved to {TestOutput_file}")


if __name__ == "__main__":
    main()