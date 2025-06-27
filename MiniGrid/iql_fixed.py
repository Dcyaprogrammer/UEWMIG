import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import flax
import flax.linen as nn
import optax
import distrax
from flax.training.train_state import TrainState
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple
from functools import partial

# 修复的 CatPolicy
class CatPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        x = observations.reshape(observations.shape[0], -1)  # flatten
        # 移除 activate_final=True，避免激活函数影响logits
        outputs = MLP(self.hidden_dims, activate_final=False)(x)
        logits = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        # 添加温度缩放
        logits = logits / temperature
        distribution = distrax.Categorical(logits=logits)
        return distribution

# 修复的数据预处理
def preprocess_dataset_fixed(dataset: dict, config, clip_to_eps: bool = True, eps: float = 1e-5):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = jnp.clip(dataset["actions"], -lim, lim)

    obs = dataset['observations']         # shape: (N, 7, 7, 2)
    next_obs = dataset['next_observations']  # shape: (N, 7, 7, 2)
    dones = dataset['dones']              # shape: (N,)

    # 简化dones_float计算，直接使用dones
    dones_float = dones.astype(jnp.float32)

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.int32),  # 改为int32
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dataset["dones"], dtype=jnp.float32),
        dones_float=jnp.array(dones_float, dtype=jnp.float32),
    )
    return dataset

# 修复的IQL更新函数
class IQLFixed:
    @classmethod
    def update_actor_fixed(
        cls, train_state, batch: Transition, config
    ) -> Tuple["IQLTrainState", Dict]:
        v = train_state.value.apply_fn(train_state.value.params, batch.observations)
        q1, q2 = train_state.critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jnp.minimum(q1, q2)
        
        # 修复exp_a计算，添加数值稳定性
        advantage = q - v
        exp_a = jnp.exp(jnp.clip(advantage * config.beta, -20, 20))  # 更保守的clip
        exp_a = jnp.minimum(exp_a, 100.0)
        
        def actor_loss_fn(actor_params):
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            # 添加数值稳定性检查
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), {"actor_loss": actor_loss, "exp_a_mean": exp_a.mean()}

    @classmethod
    def update_value_fixed(
        cls, train_state, batch: Transition, config
    ) -> Tuple["IQLTrainState", Dict]:
        q1, q2 = train_state.target_critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
        
        def value_loss_fn(value_params):
            v = train_state.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.expectile).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), {"value_loss": value_loss}

    @classmethod
    def update_critic_fixed(
        cls, train_state, batch: Transition, config
    ) -> Tuple["IQLTrainState", Dict]:
        next_v = train_state.value.apply_fn(
            train_state.value.params, batch.next_observations
        )
        target_q = batch.rewards + config.discount * (1 - batch.dones_float) * next_v

        def critic_loss_fn(critic_params):
            q1, q2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), {"critic_loss": critic_loss}

# 修复的配置参数
class IQLConfigFixed:
    # 调整超参数以提高稳定性
    expectile: float = 0.8  # 从0.7增加到0.8
    beta: float = 1.0       # 从3.0降低到1.0，减少exp_a的数值问题
    tau: float = 0.005
    discount: float = 0.99
    batch_size: int = 256
    actor_lr: float = 1e-4  # 降低学习率
    value_lr: float = 1e-4
    critic_lr: float = 1e-4
    n_jitted_updates: int = 4  # 减少更新频率

# 添加调试信息
def debug_training_info(train_state, batch, config):
    """添加调试信息来监控训练状态"""
    v = train_state.value.apply_fn(train_state.value.params, batch.observations)
    q1, q2 = train_state.critic.apply_fn(
        train_state.target_critic.params, batch.observations, batch.actions
    )
    q = jnp.minimum(q1, q2)
    advantage = q - v
    
    debug_info = {
        "v_mean": v.mean(),
        "v_std": v.std(),
        "q_mean": q.mean(),
        "q_std": q.std(),
        "advantage_mean": advantage.mean(),
        "advantage_std": advantage.std(),
        "rewards_mean": batch.rewards.mean(),
        "rewards_std": batch.rewards.std(),
    }
    return debug_info 