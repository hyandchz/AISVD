import os
import torch
from firstsolution import SVDNet, train_model
from utils.data_loader import read_cfg_file, load_training_data, get_data_paths
from config import get_device, get_case_ids

Ridx = 1


def main():
    print("<<< Starting Training Process >>>\n")

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

        # 加载训练数据
        train_data, train_label = load_training_data(PathRaw, Prefix, case_id)
        if train_data is None or train_label is None:
            print(f"No training data for Round {Ridx} Case {case_id}, skipping.")
            continue

        # 初始化模型
        model = SVDNet(M, N, R).to(device)
        model_path = f'model_round{Ridx}_case{case_id}.pth'

        # 训练模型
        print(f"Training model for Round {Ridx} Case {case_id}")
        model = train_model(model, train_data, train_label, device)

        # 保存模型
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()