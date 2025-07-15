import os
import torch
# 路径配置
PathSet = {0: "./DebugData", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}

# 当前轮次
Ridx = 1

# 设备配置
def get_device():
    return torch.device('cpu')

# 获取所有场景ID
def get_case_ids(PathRaw):
    files = os.listdir(PathRaw)
    case_ids = []
    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            case_ids.append(f.split('CfgData')[-1].split('.txt')[0])
    return case_ids