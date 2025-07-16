import os
import numpy as np
from config import PathSet, PrefixSet, Ridx


def read_cfg_file(file_path):
    """读取配置文件"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    samp_num = int(line_fmt[0][0])
    M = int(line_fmt[1][0])
    N = int(line_fmt[2][0])
    IQ = int(line_fmt[3][0])
    R = int(line_fmt[4][0])
    return samp_num, M, N, IQ, R


def load_training_data(PathRaw, Prefix, case_id):
    """加载训练数据和标签"""
    train_data_path = os.path.join(PathRaw, f'{Prefix}TrainData{case_id}.npy')
    train_label_path = os.path.join(PathRaw, f'{Prefix}TrainLabel{case_id}.npy')

    if os.path.exists(train_data_path) and os.path.exists(train_label_path):
        return np.load(train_data_path), np.load(train_label_path)
    return None, None


def load_test_data(PathRaw, Prefix, case_id):
    """加载测试数据"""
    test_data_path = os.path.join(PathRaw, f'{Prefix}TestData{case_id}.npy')
    if os.path.exists(test_data_path):
        return np.load(test_data_path)
    return None


def get_data_paths():
    """获取当前轮次的数据路径和前缀"""
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    return PathRaw, Prefix