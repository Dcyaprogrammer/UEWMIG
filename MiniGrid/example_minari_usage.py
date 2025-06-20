#!/usr/bin/env python3
"""
示例：如何使用Minari数据处理器处理EpisodeData
"""

import numpy as np
from minari_data_processor import process_episodes_directly, MinariDataProcessor
import minari


def main():
    """主函数：演示如何使用数据处理器"""
    
    # 假设您已经有了EpisodeData列表
    # 这里我们创建一个模拟的示例
    
    print("=== MiniGrid Minari数据处理示例 ===")
    
    # 1. 创建数据处理器
    processor = MinariDataProcessor(
        image_size=(22, 22),
        use_mission=False,
        normalize_image=True,
        normalize_direction=True
    )
    
    print(f"数据处理器配置:")
    print(f"  - 图像尺寸: {processor.image_size}")
    print(f"  - 图像维度: {processor.image_dim}")
    print(f"  - 方向维度: {processor.direction_dim}")
    print(f"  - 总observation维度: {processor.obs_dim}")
    
    # 2. 如果您有真实的数据集，可以这样使用：
    """
    # 加载数据集
    dataset_name = "your-minigrid-dataset-name"
    dataset = minari.load_dataset(dataset_name)
    
    # 采样episodes
    episodes = dataset.sample_episodes(1000)  # 加载1000个episodes
    
    # 处理数据
    processed_data, obs_mean, obs_std = process_episodes_directly(
        episodes=episodes,
        image_size=(22, 22),
        normalize=True
    )
    
    print(f"处理后的数据:")
    print(f"  - Observations shape: {processed_data.observations.shape}")
    print(f"  - Actions shape: {processed_data.actions.shape}")
    print(f"  - Rewards shape: {processed_data.rewards.shape}")
    print(f"  - Next observations shape: {processed_data.next_observations.shape}")
    print(f"  - Dones shape: {processed_data.dones.shape}")
    print(f"  - 总transitions: {len(processed_data.observations)}")
    
    # 3. 现在您可以将processed_data用于TD3-BC训练
    # processed_data.observations 是扁平化的observation数组
    # processed_data.actions 是动作数组
    # processed_data.rewards 是奖励数组
    # processed_data.next_observations 是下一个observation数组
    # processed_data.dones 是完成标志数组
    """
    
    # 4. 如果您想要更细粒度的控制，可以这样处理单个episode：
    """
    # 处理单个episode
    episode = episodes[0]  # 第一个episode
    transition = processor.process_episode(episode)
    
    print(f"单个episode处理结果:")
    print(f"  - Episode长度: {len(transition.observations)}")
    print(f"  - Observation shape: {transition.observations.shape}")
    print(f"  - Action shape: {transition.actions.shape}")
    """
    
    print("\n=== 使用说明 ===")
    print("1. 使用 minari.load_dataset() 加载您的数据集")
    print("2. 使用 dataset.sample_episodes() 获取EpisodeData列表")
    print("3. 调用 process_episodes_directly() 处理数据")
    print("4. 将处理后的数据传递给TD3-BC算法")
    
    print("\n=== 数据格式说明 ===")
    print("输入数据格式:")
    print("  - observations: 字典列表，每个字典包含 'image', 'direction', 'mission'")
    print("  - image: (22, 22, 3) RGB图像数组")
    print("  - direction: 方向标量 (0-3)")
    print("  - mission: 任务描述字符串")
    
    print("\n输出数据格式:")
    print("  - observations: (N, 1453) 扁平化的observation数组")
    print("  - actions: (N,) 动作数组")
    print("  - rewards: (N,) 奖励数组")
    print("  - next_observations: (N, 1453) 下一个observation数组")
    print("  - dones: (N,) 完成标志数组")
    print("  其中 N 是总transition数量，1453 = 22*22*3 + 1")


if __name__ == "__main__":
    main() 