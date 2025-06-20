import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import minari
from minari import EpisodeData


@dataclass
class ProcessedTransition:
    """处理后的transition数据"""
    observations: np.ndarray  # 扁平化的observation
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class MinariDataProcessor:
    """Minari EpisodeData处理器"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (22, 22),
                 use_mission: bool = False,
                 normalize_image: bool = True,
                 normalize_direction: bool = True):
        """
        初始化数据处理器
        
        Args:
            image_size: 图像尺寸 (H, W)
            use_mission: 是否使用mission信息
            normalize_image: 是否归一化图像到[0,1]
            normalize_direction: 是否归一化方向到[0,1]
        """
        self.image_size = image_size
        self.use_mission = use_mission
        self.normalize_image = normalize_image
        self.normalize_direction = normalize_direction
        
        # 计算observation维度
        self.image_dim = image_size[0] * image_size[1] * 3
        self.direction_dim = 1
        self.obs_dim = self.image_dim + self.direction_dim
        
        # 统计信息
        self.obs_mean = None
        self.obs_std = None
        
    def process_observation(self, obs: Dict) -> np.ndarray:
        """
        处理单个observation
        
        Args:
            obs: 包含image, direction, mission的字典
            
        Returns:
            扁平化的observation数组
        """
        obs_components = []
        
        # 处理图像
        if 'image' in obs:
            image = obs['image']
            if image.shape != self.image_size + (3,):
                raise ValueError(f"Expected image shape {self.image_size + (3,)}, got {image.shape}")
            
            # 归一化图像到[0,1]
            if self.normalize_image:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
                
            # 展平图像
            image_flat = image.flatten()
            obs_components.append(image_flat)
        
        # 处理方向
        if 'direction' in obs:
            direction = obs['direction']
            if self.normalize_direction:
                # MiniGrid中方向是0-3，归一化到[0,1]
                direction_normalized = direction.astype(np.float32) / 4.0
            else:
                direction_normalized = direction.astype(np.float32)
            
            obs_components.append(direction_normalized)
        
        # 处理mission (可选)
        if self.use_mission and 'mission' in obs:
            # 这里可以添加文本编码逻辑
            # 暂时跳过mission处理
            pass
        
        # 合并所有组件
        return np.concatenate(obs_components).astype(np.float32)
    
    def process_episode(self, episode: EpisodeData) -> ProcessedTransition:
        """
        处理单个episode
        
        Args:
            episode: Minari的EpisodeData对象
            
        Returns:
            处理后的transition数据
        """
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations
        
        # 处理observations
        processed_obs = []
        for obs in observations:
            processed_obs.append(self.process_observation(obs))
        
        # 创建next_observations (最后一个observation的next_observation设为0)
        processed_next_obs = processed_obs[1:] + [np.zeros_like(processed_obs[0])]
        
        # 计算dones (terminations或truncations)
        dones = np.logical_or(terminations, truncations).astype(np.float32)
        
        return ProcessedTransition(
            observations=np.array(processed_obs, dtype=np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            next_observations=np.array(processed_next_obs, dtype=np.float32),
            dones=dones
        )
    
    def process_episodes(self, episodes: List[EpisodeData]) -> ProcessedTransition:
        """
        处理多个episodes
        
        Args:
            episodes: EpisodeData列表
            
        Returns:
            合并后的transition数据
        """
        all_transitions = []
        
        print(f"Processing {len(episodes)} episodes...")
        for i, episode in enumerate(episodes):
            if i % 100 == 0:
                print(f"Processing episode {i}/{len(episodes)}")
            
            transition = self.process_episode(episode)
            all_transitions.append(transition)
        
        # 合并所有transitions
        all_obs = np.concatenate([t.observations for t in all_transitions], axis=0)
        all_actions = np.concatenate([t.actions for t in all_transitions], axis=0)
        all_rewards = np.concatenate([t.rewards for t in all_transitions], axis=0)
        all_next_obs = np.concatenate([t.next_observations for t in all_transitions], axis=0)
        all_dones = np.concatenate([t.dones for t in all_transitions], axis=0)
        
        return ProcessedTransition(
            observations=all_obs,
            actions=all_actions,
            rewards=all_rewards,
            next_observations=all_next_obs,
            dones=all_dones
        )
    
    def compute_statistics(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算observation的统计信息用于归一化
        
        Args:
            observations: 处理后的observation数组
            
        Returns:
            mean, std
        """
        self.obs_mean = np.mean(observations, axis=0)
        self.obs_std = np.std(observations, axis=0) + 1e-5
        return self.obs_mean, self.obs_std
    
    def normalize_observations(self, 
                             observations: np.ndarray,
                             next_observations: np.ndarray,
                             use_computed_stats: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化observations
        
        Args:
            observations: 当前observations
            next_observations: 下一个observations
            use_computed_stats: 是否使用已计算的统计信息
            
        Returns:
            归一化后的observations和next_observations
        """
        if use_computed_stats and self.obs_mean is not None and self.obs_std is not None:
            mean, std = self.obs_mean, self.obs_std
        else:
            # 使用当前数据的统计信息
            all_obs = np.concatenate([observations, next_observations], axis=0)
            mean = np.mean(all_obs, axis=0)
            std = np.std(all_obs, axis=0) + 1e-5
        
        normalized_obs = (observations - mean) / std
        normalized_next_obs = (next_observations - mean) / std
        
        return normalized_obs, normalized_next_obs


def load_and_process_minari_dataset(dataset_name: str,
                                   num_episodes: Optional[int] = None,
                                   image_size: Tuple[int, int] = (22, 22),
                                   normalize: bool = True) -> Tuple[ProcessedTransition, np.ndarray, np.ndarray]:
    """
    加载并处理Minari数据集
    
    Args:
        dataset_name: Minari数据集名称
        num_episodes: 要加载的episode数量，None表示加载全部
        image_size: 图像尺寸
        normalize: 是否归一化observations
        
    Returns:
        处理后的数据集, obs_mean, obs_std
    """
    # 加载数据集
    print(f"Loading dataset: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)
    
    # 采样episodes
    if num_episodes is not None:
        episodes = dataset.sample_episodes(num_episodes)
    else:
        episodes = dataset.sample_episodes()
    
    print(f"Loaded {len(episodes)} episodes")
    
    # 创建数据处理器
    processor = MinariDataProcessor(
        image_size=image_size,
        use_mission=False,
        normalize_image=True,
        normalize_direction=True
    )
    
    # 处理episodes
    processed_data = processor.process_episodes(episodes)
    
    print(f"Processed data shape: {processed_data.observations.shape}")
    print(f"Total transitions: {len(processed_data.observations)}")
    
    # 归一化
    obs_mean, obs_std = None, None
    if normalize:
        obs_mean, obs_std = processor.compute_statistics(processed_data.observations)
        normalized_obs, normalized_next_obs = processor.normalize_observations(
            processed_data.observations, 
            processed_data.next_observations,
            use_computed_stats=False
        )
        
        processed_data = ProcessedTransition(
            observations=normalized_obs,
            actions=processed_data.actions,
            rewards=processed_data.rewards,
            next_observations=normalized_next_obs,
            dones=processed_data.dones
        )
        
        print("Observations normalized")
    
    return processed_data, obs_mean, obs_std


def process_episodes_directly(episodes: List[EpisodeData],
                             image_size: Tuple[int, int] = (22, 22),
                             normalize: bool = True) -> Tuple[ProcessedTransition, np.ndarray, np.ndarray]:
    """
    直接处理EpisodeData列表
    
    Args:
        episodes: EpisodeData列表
        image_size: 图像尺寸
        normalize: 是否归一化observations
        
    Returns:
        处理后的数据集, obs_mean, obs_std
    """
    # 创建数据处理器
    processor = MinariDataProcessor(
        image_size=image_size,
        use_mission=False,
        normalize_image=True,
        normalize_direction=True
    )
    
    # 处理episodes
    processed_data = processor.process_episodes(episodes)
    
    print(f"Processed data shape: {processed_data.observations.shape}")
    print(f"Total transitions: {len(processed_data.observations)}")
    
    # 归一化
    obs_mean, obs_std = None, None
    if normalize:
        obs_mean, obs_std = processor.compute_statistics(processed_data.observations)
        normalized_obs, normalized_next_obs = processor.normalize_observations(
            processed_data.observations, 
            processed_data.next_observations,
            use_computed_stats=False
        )
        
        processed_data = ProcessedTransition(
            observations=normalized_obs,
            actions=processed_data.actions,
            rewards=processed_data.rewards,
            next_observations=normalized_next_obs,
            dones=processed_data.dones
        )
        
        print("Observations normalized")
    
    return processed_data, obs_mean, obs_std


# 示例使用函数
def example_usage():
    """示例：如何使用Minari数据处理器"""
    
    # 方法1：直接加载数据集
    # processed_data, obs_mean, obs_std = load_and_process_minari_dataset(
    #     dataset_name="your-dataset-name",
    #     num_episodes=1000,
    #     image_size=(22, 22),
    #     normalize=True
    # )
    
    # 方法2：处理已有的EpisodeData列表
    # dataset = minari.load_dataset("your-dataset-name")
    # episodes = dataset.sample_episodes(1000)
    # processed_data, obs_mean, obs_std = process_episodes_directly(
    #     episodes=episodes,
    #     image_size=(22, 22),
    #     normalize=True
    # )
    
    print("Minari数据处理器已创建")
    processor = MinariDataProcessor(image_size=(22, 22))
    print(f"Observation维度: {processor.obs_dim}")
    print(f"图像维度: {processor.image_dim}")
    print(f"方向维度: {processor.direction_dim}")


if __name__ == "__main__":
    example_usage() 