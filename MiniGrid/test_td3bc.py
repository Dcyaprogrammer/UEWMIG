#!/usr/bin/env python3
"""
测试修改后的TD3-BC算法
"""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from td3bc import TD3BCConfig, get_dataset, create_td3bc_train_state, TD3BC, evaluate


def process_rollout_data(transitions, actions):
    """
    处理rollout数据，构建适合offline训练的replay buffer
    
    Args:
        transitions: TimeStep对象，包含1000个时间步
        actions: 动作数组，形状为(1000,)
        
    Returns:
        replay_buffer: 包含observations, actions, rewards, next_observations, dones的字典
    """
    print("=== 处理rollout数据 ===")
    
    # 提取observations (当前状态)
    observations = transitions.observation  # (1000, 7, 7, 2)
    print(f"Observations shape: {observations.shape}")
    
    # 提取rewards
    rewards = transitions.reward  # (1000,)
    print(f"Rewards shape: {rewards.shape}")
    
    # 提取dones (episode结束标志)
    dones = transitions.step_type == 2  # LAST step_type = 2
    print(f"Dones shape: {dones.shape}")
    
    # 构建next_observations (下一个状态)
    # 对于最后一个时间步，next_observation就是当前observation
    next_observations = jnp.concatenate([
        observations[1:],  # 从第二个开始
        observations[-1:]   # 最后一个重复一次
    ], axis=0)
    print(f"Next observations shape: {next_observations.shape}")
    
    # 确保actions是正确的形状
    actions = jnp.array(actions, dtype=jnp.int32)  # (1000,)
    print(f"Actions shape: {actions.shape}")
    
    # 构建replay buffer
    replay_buffer = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'dones': dones
    }
    
    print("=== Replay Buffer 构建完成 ===")
    print(f"数据点数量: {len(observations)}")
    print(f"平均奖励: {jnp.mean(rewards):.4f}")
    print(f"Episode结束次数: {jnp.sum(dones)}")
    print(f"动作分布: {jnp.bincount(actions)}")
    
    return replay_buffer


def create_sequential_batches(replay_buffer, batch_size=32, num_batches=None, rng_key=None):
    """
    创建保持轨迹时序关系的批次
    """
    print(f"=== 创建时序批次 (batch_size={batch_size}) ===")
    
    data_size = len(replay_buffer['observations'])
    
    if num_batches is None:
        num_batches = max(1, data_size // batch_size)
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # 随机选择起始点，然后连续采样
    def generate_sequential_batch_indices(carry, _):
        rng, _ = carry
        rng, new_rng = jax.random.split(rng)
        
        # 随机选择起始点，确保有足够空间采样batch_size个连续样本
        max_start_idx = data_size - batch_size
        start_idx = jax.random.randint(new_rng, shape=(), minval=0, maxval=max_start_idx + 1)
        
        # 生成连续的索引
        batch_indices = jnp.arange(start_idx, start_idx + batch_size)
        
        return (new_rng, batch_indices), batch_indices
    
    # 使用scan生成所有批次的索引
    init_carry = (rng_key, jnp.zeros(batch_size, dtype=jnp.int32))
    _, all_batch_indices = jax.lax.scan(
        generate_sequential_batch_indices, 
        init_carry, 
        None, 
        length=num_batches
    )
    
    # 创建批次数据
    def create_batch_data(batch_indices):
        return {
            'observations': replay_buffer['observations'][batch_indices],
            'actions': replay_buffer['actions'][batch_indices],
            'rewards': replay_buffer['rewards'][batch_indices],
            'next_observations': replay_buffer['next_observations'][batch_indices],
            'dones': replay_buffer['dones'][batch_indices]
        }
    
    batches = jax.vmap(create_batch_data)(all_batch_indices)
    
    print(f"创建了 {num_batches} 个时序批次")
    print(f"每个批次包含 {batch_size} 个连续样本")
    
    return batches


def create_mixed_batches(replay_buffer, batch_size=32, num_batches=None, rng_key=None, 
                        sequential_ratio=0.5):
    """
    创建混合批次：部分保持时序关系，部分随机采样
    
    Args:
        replay_buffer: 包含训练数据的字典
        batch_size: 批次大小
        num_batches: 批次数量
        rng_key: 随机数生成器key
        sequential_ratio: 时序样本的比例 (0.0-1.0)
        
    Returns:
        batches: 批次数据字典
    """
    print(f"=== 创建混合批次 (sequential_ratio={sequential_ratio}) ===")
    
    data_size = len(replay_buffer['observations'])
    
    if num_batches is None:
        num_batches = max(1, data_size // batch_size)
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # 计算时序样本和随机样本的数量
    sequential_size = int(batch_size * sequential_ratio)
    random_size = batch_size - sequential_size
    
    def generate_mixed_batch_indices(carry, _):
        rng, _ = carry
        rng, seq_rng, rand_rng = jax.random.split(rng, 3)
        
        # 生成时序索引
        max_start_idx = data_size - sequential_size
        start_idx = jax.random.randint(seq_rng, shape=(), minval=0, maxval=max_start_idx + 1)
        sequential_indices = jnp.arange(start_idx, start_idx + sequential_size)
        
        # 生成随机索引
        random_indices = jax.random.randint(
            rand_rng, 
            shape=(random_size,), 
            minval=0, 
            maxval=data_size
        )
        
        # 合并索引
        batch_indices = jnp.concatenate([sequential_indices, random_indices])
        
        return (rng, batch_indices), batch_indices
    
    # 使用scan生成所有批次的索引
    init_carry = (rng_key, jnp.zeros(batch_size, dtype=jnp.int32))
    _, all_batch_indices = jax.lax.scan(
        generate_mixed_batch_indices, 
        init_carry, 
        None, 
        length=num_batches
    )
    
    # 创建批次数据
    def create_batch_data(batch_indices):
        return {
            'observations': replay_buffer['observations'][batch_indices],
            'actions': replay_buffer['actions'][batch_indices],
            'rewards': replay_buffer['rewards'][batch_indices],
            'next_observations': replay_buffer['next_observations'][batch_indices],
            'dones': replay_buffer['dones'][batch_indices]
        }
    
    batches = jax.vmap(create_batch_data)(all_batch_indices)
    
    print(f"创建了 {num_batches} 个混合批次")
    print(f"每个批次包含 {sequential_size} 个时序样本 + {random_size} 个随机样本")
    
    return batches


def create_episode_aware_batches(replay_buffer, episode_boundaries, batch_size=32, 
                                num_batches=None, rng_key=None):
    """
    创建episode感知的批次：确保批次内的样本来自同一个episode
    
    Args:
        replay_buffer: 包含训练数据的字典
        episode_boundaries: episode边界列表，例如 [0, 100, 250, 400, 1000]
        batch_size: 批次大小
        num_batches: 批次数量
        rng_key: 随机数生成器key
        
    Returns:
        batches: 批次数据字典
    """
    print(f"=== 创建Episode感知批次 ===")
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # 计算每个episode的长度
    episode_lengths = []
    for i in range(len(episode_boundaries) - 1):
        start = episode_boundaries[i]
        end = episode_boundaries[i + 1]
        episode_lengths.append(end - start)
    
    print(f"Episode数量: {len(episode_lengths)}")
    print(f"Episode长度: {episode_lengths}")
    
    if num_batches is None:
        num_batches = max(1, sum(episode_lengths) // batch_size)
    
    def generate_episode_batch_indices(carry, _):
        rng, _ = carry
        rng, episode_rng, start_rng = jax.random.split(rng, 3)
        
        # 随机选择一个episode
        episode_idx = jax.random.randint(episode_rng, shape=(), minval=0, maxval=len(episode_lengths))
        
        # 获取episode的起始和结束位置
        episode_start = episode_boundaries[episode_idx]
        episode_end = episode_boundaries[episode_idx + 1]
        episode_length = episode_end - episode_start
        
        # 如果episode太短，使用整个episode
        if episode_length <= batch_size:
            batch_indices = jnp.arange(episode_start, episode_end)
            # 如果不够batch_size，用最后一个样本填充
            if episode_length < batch_size:
                padding = jnp.full(batch_size - episode_length, episode_end - 1)
                batch_indices = jnp.concatenate([batch_indices, padding])
        else:
            # 在episode内随机选择起始点
            max_start = episode_start + episode_length - batch_size
            start_idx = jax.random.randint(start_rng, shape=(), minval=episode_start, maxval=max_start + 1)
            batch_indices = jnp.arange(start_idx, start_idx + batch_size)
        
        return (rng, batch_indices), batch_indices
    
    # 使用scan生成所有批次的索引
    init_carry = (rng_key, jnp.zeros(batch_size, dtype=jnp.int32))
    _, all_batch_indices = jax.lax.scan(
        generate_episode_batch_indices, 
        init_carry, 
        None, 
        length=num_batches
    )
    
    # 创建批次数据
    def create_batch_data(batch_indices):
        return {
            'observations': replay_buffer['observations'][batch_indices],
            'actions': replay_buffer['actions'][batch_indices],
            'rewards': replay_buffer['rewards'][batch_indices],
            'next_observations': replay_buffer['next_observations'][batch_indices],
            'dones': replay_buffer['dones'][batch_indices]
        }
    
    batches = jax.vmap(create_batch_data)(all_batch_indices)
    
    print(f"创建了 {num_batches} 个Episode感知批次")
    
    return batches


def analyze_sampling_strategies():
    """
    分析不同采样策略的优缺点
    """
    print("\n=== 采样策略分析 ===")
    
    strategies = {
        "完全随机采样": {
            "优点": [
                "数据多样性高",
                "避免过拟合特定轨迹",
                "实现简单",
                "内存效率高"
            ],
            "缺点": [
                "丢失时序关系",
                "可能影响状态转移学习",
                "不适合需要理解轨迹结构的任务"
            ],
            "适用场景": [
                "行为克隆 (BC)",
                "价值函数学习",
                "策略梯度方法"
            ]
        },
        "时序采样": {
            "优点": [
                "保持时序关系",
                "有利于状态转移学习",
                "符合环境动态",
                "适合序列建模"
            ],
            "缺点": [
                "数据多样性可能降低",
                "可能过拟合特定轨迹模式",
                "实现相对复杂"
            ],
            "适用场景": [
                "模型学习 (World Model)",
                "序列预测",
                "需要理解因果关系的任务"
            ]
        },
        "混合采样": {
            "优点": [
                "平衡时序性和多样性",
                "灵活性高",
                "可以根据任务调整比例"
            ],
            "缺点": [
                "需要调参",
                "实现复杂"
            ],
            "适用场景": [
                "通用强化学习",
                "需要平衡探索和利用的任务"
            ]
        },
        "Episode感知采样": {
            "优点": [
                "保持episode完整性",
                "有利于学习episode级别的模式",
                "适合多步规划"
            ],
            "缺点": [
                "需要episode边界信息",
                "可能限制数据使用效率"
            ],
            "适用场景": [
                "多步规划",
                "需要理解episode结构的任务"
            ]
        }
    }
    
    for strategy, info in strategies.items():
        print(f"\n{strategy}:")
        print("  优点:", ", ".join(info["优点"]))
        print("  缺点:", ", ".join(info["缺点"]))
        print("  适用场景:", ", ".join(info["适用场景"]))


def test_rollout_data_processing():
    """测试rollout数据处理功能"""
    print("=== 测试rollout数据处理 ===")
    
    # 模拟你的rollout数据（这里用随机数据代替）
    # 在实际使用中，你会传入真实的rollout结果
    
    # 创建模拟的transitions
    from typing import NamedTuple
    
    class MockState(NamedTuple):
        key: jnp.ndarray
        step_num: jnp.ndarray
        grid: jnp.ndarray
        agent: jnp.ndarray
        goal_encoding: jnp.ndarray
        rule_encoding: jnp.ndarray
        carry: jnp.ndarray
    
    class MockTimeStep(NamedTuple):
        state: MockState
        step_type: jnp.ndarray
        reward: jnp.ndarray
        discount: jnp.ndarray
        observation: jnp.ndarray
    
    # 创建模拟数据
    rng = jax.random.PRNGKey(42)
    rng, obs_rng, action_rng, reward_rng, step_type_rng = jax.random.split(rng, 5)
    
    # 模拟1000个时间步的数据
    num_steps = 1000
    
    # 创建模拟的TimeStep
    mock_transitions = MockTimeStep(
        state=MockState(
            key=jax.random.randint(rng, (num_steps,), 0, 1000),
            step_num=jnp.arange(num_steps),
            grid=jax.random.randint(rng, (num_steps, 8, 8, 2), 0, 10),
            agent=jax.random.randint(rng, (num_steps, 3), 0, 8),
            goal_encoding=jax.random.randint(rng, (num_steps, 5), 0, 2),
            rule_encoding=jax.random.randint(rng, (num_steps, 1, 7), 0, 2),
            carry=jnp.zeros((num_steps,))
        ),
        step_type=jax.random.randint(step_type_rng, (num_steps,), 0, 3),  # 0, 1, 2
        reward=jax.random.uniform(reward_rng, (num_steps,), -1.0, 1.0),
        discount=jnp.ones((num_steps,)) * 0.99,
        observation=jax.random.uniform(obs_rng, (num_steps, 7, 7, 2), 0.0, 1.0)
    )
    
    # 创建模拟的actions
    mock_actions = jax.random.randint(action_rng, (num_steps,), 0, 7)
    
    # 处理数据
    replay_buffer = process_rollout_data(mock_transitions, mock_actions)
    
    # 创建训练批次 - 测试不同的采样策略
    print("\n=== 测试不同采样策略 ===")
    
    # 方法1: 完全随机采样 (原始方法)
    print("\n1. 完全随机采样:")
    batches_random = create_training_batches(replay_buffer, batch_size=32)
    
    # 方法2: 时序采样
    print("\n2. 时序采样:")
    batches_sequential = create_sequential_batches(replay_buffer, batch_size=32)
    
    # 比较两种方法
    print("\n=== 采样策略比较 ===")
    
    # 检查随机采样的时序关系
    random_batch = batches_random[0]
    random_obs = random_batch['observations']
    random_next_obs = random_batch['next_observations']
    
    # 检查时序采样的时序关系
    sequential_batch = batches_sequential[0]
    sequential_obs = sequential_batch['observations']
    sequential_next_obs = sequential_batch['next_observations']
    
    print("随机采样 - 前5个样本的索引:")
    print(f"Observations: {random_obs[:5]}")
    print(f"Next observations: {random_next_obs[:5]}")
    
    print("\n时序采样 - 前5个样本的索引:")
    print(f"Observations: {sequential_obs[:5]}")
    print(f"Next observations: {sequential_next_obs[:5]}")
    
    return replay_buffer, batches_random, batches_sequential


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


def offline_training_with_rollout_data():
    """使用rollout数据进行offline训练"""
    print("=== 使用rollout数据进行offline训练 ===")
    
    # 1. 创建环境并收集rollout数据
    import xminigrid
    from xminigrid.wrappers import GymAutoResetWrapper
    
    def build_rollout(env, env_params, num_steps):
        def rollout(rng):
            def _step_fn(carry, _):
                rng, timestep = carry
                rng, _rng = jax.random.split(rng)
                action = jax.random.randint(_rng, shape=(), minval=0, maxval=env.num_actions(env_params))
                
                timestep = env.step(env_params, timestep, action)
                
                return (rng, timestep), (timestep, action)
            
            rng, _rng = jax.random.split(rng)
            timestep = env.reset(env_params, _rng)
            rng, (transitions, actions) = jax.lax.scan(_step_fn, (rng, timestep), None, length=num_steps)
            
            return transitions, actions
        return rollout
    
    # 创建环境
    env, env_params = xminigrid.make("MiniGrid-Empty-8x8")
    env = GymAutoResetWrapper(env)
    
    # 收集rollout数据
    rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=1000))
    transitions, actions = rollout_fn(jax.random.key(42))
    
    print("原始数据形状:")
    print(f"Transitions shapes: {jax.tree_util.tree_map(jnp.shape, transitions)}")
    print(f"Actions shape: {actions.shape}")
    
    # 2. 处理数据构建replay buffer
    replay_buffer = process_rollout_data(transitions, actions)
    
    # 3. 创建TD3-BC配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        batch_size=32,
        n_jitted_updates=4,
        max_steps=10000,  # 训练步数
        data_size=len(replay_buffer['observations']),
        normalize_state=True
    )
    
    # 4. 创建训练状态
    rng = jax.random.PRNGKey(42)
    
    # 获取一个样本batch来初始化网络
    sample_batch = {
        'observations': replay_buffer['observations'][:config.batch_size],
        'actions': replay_buffer['actions'][:config.batch_size],
        'rewards': replay_buffer['rewards'][:config.batch_size],
        'next_observations': replay_buffer['next_observations'][:config.batch_size],
        'dones': replay_buffer['dones'][:config.batch_size]
    }
    
    # 计算observation的统计信息用于归一化
    obs_mean = jnp.mean(replay_buffer['observations'], axis=0)
    obs_std = jnp.std(replay_buffer['observations'], axis=0) + 1e-8
    
    # 归一化observations
    normalized_obs = (replay_buffer['observations'] - obs_mean) / obs_std
    normalized_next_obs = (replay_buffer['next_observations'] - obs_mean) / obs_std
    
    # 创建Transition对象
    from td3bc import Transition
    dataset = Transition(
        observations=normalized_obs,
        actions=replay_buffer['actions'].astype(jnp.float32),  # TD3-BC期望连续动作
        rewards=replay_buffer['rewards'],
        next_observations=normalized_next_obs,
        dones=replay_buffer['dones']
    )
    
    # 创建训练状态
    train_state = create_td3bc_train_state(
        rng, 
        dataset.observations[:config.batch_size], 
        dataset.actions[:config.batch_size], 
        config
    )
    
    print(f"训练状态创建完成")
    print(f"Actor参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(train_state.actor.params))}")
    print(f"Critic参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(train_state.critic.params))}")
    
    # 5. 开始训练
    algo = TD3BC()
    
    print("开始训练...")
    for step in range(0, config.max_steps, config.n_jitted_updates):
        rng, update_rng = jax.random.split(rng)
        
        # 执行更新
        train_state, update_info = algo.update_n_times(
            train_state, dataset, update_rng, config
        )
        
        # 打印训练信息
        if step % 1000 == 0:
            print(f"Step {step}: Critic Loss = {update_info['critic_loss']:.4f}, "
                  f"Actor Loss = {update_info['actor_loss']:.4f}")
    
    print("训练完成!")
    
    # 6. 评估训练结果
    print("评估训练结果...")
    
    def policy_fn(obs):
        # 归一化observation
        normalized_obs = (obs - obs_mean) / obs_std
        return algo.get_action(train_state=train_state, obs=normalized_obs)
    
    score = evaluate(
        policy_fn,
        config.env_name,
        num_episodes=5,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_steps_per_episode=100
    )
    
    print(f"评估分数: {score:.4f}")
    
    return train_state, score


def simple_rollout_training_example():
    """简单的rollout训练示例"""
    print("=== 简单rollout训练示例 ===")
    
    # 假设你已经有了rollout数据
    # transitions: TimeStep对象，包含1000个时间步
    # actions: 动作数组，形状为(1000,)
    
    # 1. 处理你的rollout数据
    def process_your_rollout_data(transitions, actions):
        """处理你的rollout数据"""
        # 提取observations
        observations = transitions.observation  # (1000, 7, 7, 2)
        
        # 提取rewards
        rewards = transitions.reward  # (1000,)
        
        # 提取dones (episode结束标志)
        dones = transitions.step_type == 2  # LAST step_type = 2
        
        # 构建next_observations
        next_observations = jnp.concatenate([
            observations[1:],  # 从第二个开始
            observations[-1:]   # 最后一个重复一次
        ], axis=0)
        
        # 确保actions是正确的形状
        actions = jnp.array(actions, dtype=jnp.int32)  # (1000,)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones
        }
    
    # 2. 创建训练配置
    config = TD3BCConfig(
        env_name="MiniGrid-Empty-8x8",
        batch_size=32,
        n_jitted_updates=4,
        max_steps=5000,  # 训练步数
        normalize_state=True
    )
    
    # 3. 模拟你的rollout数据（在实际使用中替换为真实数据）
    print("模拟rollout数据...")
    
    # 创建模拟的transitions和actions
    rng = jax.random.PRNGKey(42)
    rng, obs_rng, action_rng, reward_rng, step_type_rng = jax.random.split(rng, 5)
    
    num_steps = 1000
    
    # 模拟TimeStep结构
    from typing import NamedTuple
    
    class MockState(NamedTuple):
        key: jnp.ndarray
        step_num: jnp.ndarray
        grid: jnp.ndarray
        agent: jnp.ndarray
        goal_encoding: jnp.ndarray
        rule_encoding: jnp.ndarray
        carry: jnp.ndarray
    
    class MockTimeStep(NamedTuple):
        state: MockState
        step_type: jnp.ndarray
        reward: jnp.ndarray
        discount: jnp.ndarray
        observation: jnp.ndarray
    
    # 创建模拟数据
    mock_transitions = MockTimeStep(
        state=MockState(
            key=jax.random.randint(rng, (num_steps,), 0, 1000),
            step_num=jnp.arange(num_steps),
            grid=jax.random.randint(rng, (num_steps, 8, 8, 2), 0, 10),
            agent=jax.random.randint(rng, (num_steps, 3), 0, 8),
            goal_encoding=jax.random.randint(rng, (num_steps, 5), 0, 2),
            rule_encoding=jax.random.randint(rng, (num_steps, 1, 7), 0, 2),
            carry=jnp.zeros((num_steps,))
        ),
        step_type=jax.random.randint(step_type_rng, (num_steps,), 0, 3),
        reward=jax.random.uniform(reward_rng, (num_steps,), -1.0, 1.0),
        discount=jnp.ones((num_steps,)) * 0.99,
        observation=jax.random.uniform(obs_rng, (num_steps, 7, 7, 2), 0.0, 1.0)
    )
    
    mock_actions = jax.random.randint(action_rng, (num_steps,), 0, 7)
    
    # 4. 处理数据
    replay_buffer = process_your_rollout_data(mock_transitions, mock_actions)
    
    print(f"Replay buffer 大小: {len(replay_buffer['observations'])}")
    print(f"Observations shape: {replay_buffer['observations'].shape}")
    print(f"Actions shape: {replay_buffer['actions'].shape}")
    
    # 5. 数据归一化
    obs_mean = jnp.mean(replay_buffer['observations'], axis=0)
    obs_std = jnp.std(replay_buffer['observations'], axis=0) + 1e-8
    
    normalized_obs = (replay_buffer['observations'] - obs_mean) / obs_std
    normalized_next_obs = (replay_buffer['next_observations'] - obs_mean) / obs_std
    
    # 6. 创建Transition对象
    from td3bc import Transition
    dataset = Transition(
        observations=normalized_obs,
        actions=replay_buffer['actions'].astype(jnp.float32),  # TD3-BC期望连续动作
        rewards=replay_buffer['rewards'],
        next_observations=normalized_next_obs,
        dones=replay_buffer['dones']
    )
    
    # 7. 创建训练状态
    rng = jax.random.PRNGKey(42)
    train_state = create_td3bc_train_state(
        rng, 
        dataset.observations[:config.batch_size], 
        dataset.actions[:config.batch_size], 
        config
    )
    
    # 8. 开始训练
    algo = TD3BC()
    
    print("开始训练...")
    for step in range(0, config.max_steps, config.n_jitted_updates):
        rng, update_rng = jax.random.split(rng)
        
        # 执行更新
        train_state, update_info = algo.update_n_times(
            train_state, dataset, update_rng, config
        )
        
        # 打印训练信息
        if step % 1000 == 0:
            print(f"Step {step}: Critic Loss = {update_info['critic_loss']:.4f}, "
                  f"Actor Loss = {update_info['actor_loss']:.4f}")
    
    print("训练完成!")
    
    return train_state, obs_mean, obs_std


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
        
        # 测试rollout数据处理
        replay_buffer, batches_random, batches_sequential = test_rollout_data_processing()
        
        # 测试offline训练
        print("\n" + "="*50)
        print("开始offline训练测试...")
        print("="*50)
        offline_training_with_rollout_data()
        
        # 测试simple_rollout_training_example
        print("\n" + "="*50)
        print("开始simple_rollout_training_example测试...")
        print("="*50)
        simple_rollout_training_example()
        
        print("\n=== 所有测试通过！ ===")
        print("TD3-BC算法已成功从gym/d4rl迁移到xminigrid")
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 