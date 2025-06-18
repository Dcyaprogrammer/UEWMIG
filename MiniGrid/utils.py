import minari
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# 创建颜色和类型的映射字典
COLOR_MAP = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5
}

TYPE_MAP = {
    "ball": 0,
    "box": 1,
    "key": 2
}

def mission_to_int(mission) -> int:
    """
    将单个mission字符串转换为整数
    Args:
        mission: 格式为 "pick up a {color} {type}" 的字符串
    Returns:
        整数编码，格式为 color_id * 10 + type_id
    """
    # 分割字符串获取颜色和类型
    parts = mission.split()
    color = parts[3]  # 获取颜色
    obj_type = parts[4]  # 获取类型
    
    # 获取对应的ID
    color_id = COLOR_MAP[color]
    type_id = TYPE_MAP[obj_type]
    
    # 组合成唯一整数 (使用color_id * 10 + type_id确保唯一性)
    return color_id * 10 + type_id

def encode_missions(missions) -> np.ndarray:
    """
    将mission列表转换为整数数组
    Args:
        missions: mission字符串列表
    Returns:
        整数数组，每个元素对应一个mission的编码
    """
    return np.array([mission_to_int(mission) for mission in missions])


dataset = minari.load_dataset(
    dataset_id="minigrid/BabyAI-Pickup/optimal-fullobs-v0",
)






