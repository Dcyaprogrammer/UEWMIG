#!/usr/bin/env python3
"""
调试xminigrid的observation结构
"""

import jax
import jax.numpy as jnp
import numpy as np
import xminigrid

def debug_observation():
    """调试observation结构"""
    print("=== 调试xminigrid observation结构 ===")
    
    # 创建环境
    env, env_params = xminigrid.make("MiniGrid-Empty-8x8")
    
    print(f"环境名称: MiniGrid-Empty-8x8")
    print(f"Observation shape: {env.observation_shape(env_params)}")
    print(f"Num actions: {env.num_actions(env_params)}")
    
    # 重置环境
    rng = jax.random.PRNGKey(42)
    timestep = env.reset(env_params, rng)
    
    print(f"\nTimestep类型: {type(timestep)}")
    print(f"Timestep属性: {dir(timestep)}")
    
    # 检查observation
    obs = timestep.observation
    print(f"\nObservation类型: {type(obs)}")
    print(f"Observation形状: {obs.shape}")
    print(f"Observation数据类型: {obs.dtype}")
    
    # 转换为numpy查看内容
    obs_numpy = np.array(obs)
    print(f"Observation numpy形状: {obs_numpy.shape}")
    print(f"Observation内容示例:")
    print(obs_numpy)
    
    # 执行一步
    next_timestep = env.step(env_params, timestep, 0)
    next_obs = next_timestep.observation
    print(f"\nNext observation形状: {next_obs.shape}")
    
    # 检查是否包含方向信息
    if hasattr(timestep, 'agent_dir'):
        print(f"Agent direction: {timestep.agent_dir}")
    
    # 尝试渲染
    try:
        rendered = env.render(env_params, timestep)
        print(f"Rendered形状: {rendered.shape}")
    except Exception as e:
        print(f"渲染失败: {e}")

if __name__ == "__main__":
    debug_observation() 