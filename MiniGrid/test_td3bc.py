#!/usr/bin/env python3
"""
测试修改后的TD3-BC算法
"""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from td3bc import TD3BCConfig, get_dataset, create_td3bc_train_state, TD3BC, evaluate


def test_data_generation():
    """测试数据生成功能"""
    print("=== 测试数据生成 ===")
    
    # 创建配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        num_episodes=10,  # 少量episode用于测试
        data_size=1000,
        normalize_state=True
    )
    
    # 获取数据集
    dataset, obs_mean, obs_std = get_dataset(config)
    
    print(f"数据集大小: {len(dataset.observations)}")
    print(f"Observation形状: {dataset.observations.shape}")
    print(f"Action形状: {dataset.actions.shape}")
    print(f"Reward形状: {dataset.rewards.shape}")
    print(f"Next observation形状: {dataset.next_observations.shape}")
    print(f"Done形状: {dataset.dones.shape}")
    print(f"Observation均值: {obs_mean}")
    print(f"Observation标准差: {obs_std}")
    
    return dataset, obs_mean, obs_std


def test_network_creation():
    """测试网络创建"""
    print("\n=== 测试网络创建 ===")
    
    # 创建配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        num_episodes=10,
        data_size=1000
    )
    
    # 获取数据集
    dataset, _, _ = get_dataset(config)
    
    # 创建训练状态
    rng = jax.random.PRNGKey(42)
    example_batch = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_td3bc_train_state(
        rng, example_batch.observations, example_batch.actions, config
    )
    
    print(f"Actor参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(train_state.actor.params))}")
    print(f"Critic参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(train_state.critic.params))}")
    
    return train_state


def test_training_step():
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")
    
    # 创建配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        num_episodes=10,
        data_size=1000,
        batch_size=32,
        n_jitted_updates=2
    )
    
    # 获取数据集和训练状态
    dataset, _, _ = get_dataset(config)
    train_state = test_network_creation()
    
    # 测试训练步骤
    algo = TD3BC()
    rng = jax.random.PRNGKey(42)
    
    # 执行一次更新
    new_train_state, update_info = algo.update_n_times(
        train_state, dataset, rng, config
    )
    
    print(f"更新信息: {update_info}")
    print("训练步骤测试完成")


def test_evaluation():
    """测试评估功能"""
    print("\n=== 测试评估功能 ===")
    
    # 创建配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        num_episodes=2,  # 少量episode用于测试
        data_size=1000
    )
    
    # 获取数据集和训练状态
    dataset, obs_mean, obs_std = get_dataset(config)
    train_state = test_network_creation()
    
    # 测试评估
    algo = TD3BC()
    act_fn = jax.jit(algo.get_action)
    
    def policy_fn(obs):
        return act_fn(train_state=train_state, obs=obs)
    
    score = evaluate(
        policy_fn,
        config.env_name,
        num_episodes=2,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_steps_per_episode=50
    )
    
    print(f"评估分数: {score}")
    print("评估测试完成")


def main():
    """主测试函数"""
    print("开始测试修改后的TD3-BC算法...")
    
    try:
        # 测试数据生成
        dataset, obs_mean, obs_std = test_data_generation()
        
        # 测试网络创建
        train_state = test_network_creation()
        
        # 测试训练步骤
        test_training_step()
        
        # 测试评估
        test_evaluation()
        
        print("\n=== 所有测试通过！ ===")
        print("TD3-BC算法已成功从gym/d4rl迁移到xminigrid")
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 