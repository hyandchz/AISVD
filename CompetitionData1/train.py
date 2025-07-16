import os
import numpy as np
import torch
from basesolution import SVDNet, train_model
from utils.data_loader import read_cfg_file, load_training_data, get_data_paths
from config import get_device, get_case_ids


def main():
    print("<<< Starting Training Process >>>\n")

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    # 设置数据轮次为1（CompetitionData1）
    round_idx = 1

    # 获取CompetitionData1的路径
    PathRaw, Prefix = get_data_paths()
    print(f"Data directory: {PathRaw}")
    print(f"File prefix: {Prefix}")

    # 获取CompetitionData1中的所有场景ID
    case_ids = get_case_ids(PathRaw)
    print(f"Found case IDs: {case_ids}")

    # 存储所有训练数据和标签
    all_train_data = []
    all_train_label = []
    config_params = []  # 存储每组数据的配置参数

    # 处理当前轮次的每个场景
    for case_id in case_ids:
        print(f'\nProcessing Round {round_idx} Case {case_id}')

        # 读取配置文件
        cfg_path = os.path.join(PathRaw, f'{Prefix}CfgData{case_id}.txt')
        samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
        config_params.append((M, N, R))  # 保存配置参数
        print(f"Configuration: M={M}, N={N}, R={R}")

        # 加载训练数据
        train_data, train_label = load_training_data(PathRaw, Prefix, case_id)
        if train_data is None or train_label is None:
            print(f"No training data for Round {round_idx} Case {case_id}, skipping.")
            continue

        print(f"Loaded training data: {train_data.shape}")
        print(f"Loaded training labels: {train_label.shape}")

        # 添加到训练数据列表
        all_train_data.append(train_data)
        all_train_label.append(train_label)

    # 检查是否成功加载数据
    if not all_train_data:
        print("No training data found. Exiting.")
        return

    # 检查所有场景的配置是否一致
    if len(set(config_params)) > 1:
        print("Warning: Different configurations found across cases:")
        for i, (M, N, R) in enumerate(config_params):
            print(f"Case {i + 1}: M={M}, N={N}, R={R}")

        # 使用第一个场景的配置初始化模型
        M, N, R = config_params[0]
        print(f"Using configuration from Case 1 for model initialization")
    else:
        M, N, R = config_params[0]

    # 初始化模型
    model = SVDNet(M, N, R).to(device)
    print(f"\nModel initialized with M={M}, N={N}, R={R}")

    # 训练模型（使用所有场景数据）
    model_path = f'round{round_idx}_final_model.pth'
    print(f"\nStarting multi-scene training with {len(all_train_data)} datasets...")
    model = train_model(
        model,
        all_train_data,
        all_train_label,
        device,
        epochs_per_round=50,
        batch_size=128,
        patience=5,
        model_save_path='final_model.pth'
    )
    print(f"\nTraining completed. Final model saved to {model_path}")


if __name__ == "__main__":
    main()