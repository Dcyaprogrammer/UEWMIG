# source https://github.com/sfujim/TD3_BC
# https://arxiv.org/abs/2106.06860
import os
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel

import xminigrid


# 导入数据处理相关
from minari_data_processor import MinariDataProcessor, ProcessedTransition, load_and_process_minari_dataset

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


class TD3BCConfig(BaseModel):
    # GENERAL
    algo: str = "TD3-BC"
    project: str = "train-TD3-BC"
    env_name: str = "MiniGrid-Empty-8x8"
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 100000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_jitted_updates: int = 8
    # DATASET
    data_size: int = int(1e6)
    normalize_state: bool = True
    # 新增：数据集相关配置
    dataset_name: Optional[str] = None  # Minari数据集名称，如果为None则生成数据
    num_episodes: Optional[int] = 1000  # 生成数据时的episode数量
    # NETWORK
    hidden_dims: Sequence[int] = (256, 256)
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC SPECIFIC
    policy_freq: int = 2  # update actor every policy_freq updates
    alpha: float = 2.5  # BC loss weight
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    tau: float = 0.005  # target network update rate
    discount: float = 0.99  # discount factor

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = TD3BCConfig(**conf_dict)


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:  # Add layer norm after activation
                    if i + 1 < len(self.hidden_dims):
                        x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), layer_norm=True)(x)
        return q1, q2


class TD3Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0  # In D4RL, action is scaled to [-1, 1]

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        action = self.max_action * jnp.tanh(
            action
        )  # scale to [-max_action, max_action]
        return action


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray



def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class TD3BCTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    max_action: float = 1.0


class TD3BC(object):
    @classmethod
    def update_actor(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            predicted_action = train_state.actor.apply_fn(
                actor_params, batch.observations
            )
            critic_params = jax.lax.stop_gradient(train_state.critic.params)
            q_value, _ = train_state.critic.apply_fn(
                critic_params, batch.observations, predicted_action
            )

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
            loss_lambda = config.alpha / mean_abs_q

            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
            return loss_actor

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_critic(
        self,
        train_state: TD3BCTrainState,
        batch: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", jnp.ndarray]:
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q_pred_1, q_pred_2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            target_next_action = train_state.target_actor.apply_fn(
                train_state.target_actor.params, batch.next_observations
            )
            policy_noise = (
                config.policy_noise_std
                * train_state.max_action
                * jax.random.normal(rng, batch.actions.shape)
            )
            target_next_action = target_next_action + policy_noise.clip(
                -config.policy_noise_clip, config.policy_noise_clip
            )
            target_next_action = target_next_action.clip(
                -train_state.max_action, train_state.max_action
            )
            q_next_1, q_next_2 = train_state.target_critic.apply_fn(
                train_state.target_critic.params,
                batch.next_observations,
                target_next_action,
            )
            target = batch.rewards[..., None] + config.discount * jnp.minimum(
                q_next_1, q_next_2
            ) * (1 - batch.dones[..., None])
            target = jax.lax.stop_gradient(target)  # stop gradient for target
            value_loss_1 = jnp.square(q_pred_1 - target)
            value_loss_2 = jnp.square(q_pred_2 - target)
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_n_times(
        self,
        train_state: TD3BCTrainState,
        data: Transition,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple["TD3BCTrainState", Dict]:
        for _ in range(
            config.n_jitted_updates
        ):  # we can jit for roop for static unroll
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_util.tree_map(lambda x: x[batch_idx], data)
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)
            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )
            if _ % config.policy_freq == 0:
                train_state, actor_loss = self.update_actor(
                    train_state, batch, actor_rng, config
                )
                new_target_critic = target_update(
                    train_state.critic, train_state.target_critic, config.tau
                )
                new_target_actor = target_update(
                    train_state.actor, train_state.target_actor, config.tau
                )
                train_state = train_state._replace(
                    target_critic=new_target_critic,
                    target_actor=new_target_actor,
                )
        return train_state, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: TD3BCTrainState,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        action = train_state.actor.apply_fn(train_state.actor.params, obs)
        action = action.clip(-max_action, max_action)
        return action


def create_td3bc_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: TD3BCConfig,
) -> TD3BCTrainState:
    critic_model = DoubleCritic(
        hidden_dims=config.hidden_dims,
    )
    action_dim = actions.shape[-1]
    actor_model = TD3Actor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)
    # initialize critic
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(config.critic_lr),
    )
    # initialize actor
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(config.actor_lr),
    )
    return TD3BCTrainState(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
    )


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env_name: str,
    num_episodes: int,
    obs_mean,
    obs_std,
    max_steps_per_episode: int = 100,
) -> float:
    """
    评估策略
    
    Args:
        policy_fn: 策略函数
        env_name: 环境名称
        num_episodes: episode数量
        obs_mean: observation均值
        obs_std: observation标准差
        max_steps_per_episode: 每个episode的最大步数
        
    Returns:
        平均episode回报
    """
    # 创建环境
    env, env_params = xminigrid.make(env_name)
    
    # 创建数据处理器
    processor = MinariDataProcessor(
        image_size=(22, 22),
        use_mission=False,
        normalize_image=True,
        normalize_direction=True
    )
    
    episode_returns = []
    
    for episode in range(num_episodes):
        episode_return = 0
        timestep = env.reset(env_params, jax.random.PRNGKey(episode))
        
        for step in range(max_steps_per_episode):
            # 处理observation - xminigrid的observation是直接的JAX数组
            obs_array = timestep.observation
            obs_numpy = np.array(obs_array)
            
            # xminigrid的observation形状是(7, 7, 2)
            if obs_numpy.shape == (7, 7, 2):
                # 将(7, 7, 2)转换为(7, 7, 3)的RGB图像
                object_types = obs_numpy[:, :, 0]
                colors = obs_numpy[:, :, 1]
                
                rgb_image = np.zeros((7, 7, 3), dtype=np.uint8)
                rgb_image[:, :, 0] = colors
                rgb_image[:, :, 1] = object_types
                rgb_image[:, :, 2] = 0
                
                # 上采样到22x22
                from scipy.ndimage import zoom
                try:
                    rgb_image = zoom(rgb_image, (22/7, 22/7, 1), order=0)
                except ImportError:
                    rgb_image = np.repeat(np.repeat(rgb_image, 3, axis=0), 3, axis=1)
                    rgb_image = rgb_image[:22, :22, :]
                
                direction = np.array([0.0])
            else:
                rgb_image = obs_numpy
                direction = np.array([0.0])
            
            obs_dict = {
                'image': rgb_image,
                'direction': direction
            }
            processed_obs = processor.process_observation(obs_dict)
            
            # 归一化observation
            if obs_mean is not None and obs_std is not None:
                processed_obs = (processed_obs - obs_mean) / obs_std
            
            # 获取动作
            action = policy_fn(obs=processed_obs)
            
            # 执行动作
            timestep = env.step(env_params, timestep, action)
            episode_return += timestep.reward
            
            if timestep.is_done():
                break
        
        episode_returns.append(episode_return)
    
    return np.mean(episode_returns)


if __name__ == "__main__":
    wandb.init(project=config.project, config=config)
    
    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_dataset(config)
    
    # create train_state
    rng, subkey = jax.random.split(rng)
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_td3bc_train_state(
        subkey, example_batch.observations, example_batch.actions, config
    )
    algo = TD3BC()
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)

    num_steps = config.max_steps // config.n_jitted_updates
    eval_interval = config.eval_interval // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, update_rng = jax.random.split(rng)
        train_state, update_info = update_fn(
            train_state,
            dataset,
            update_rng,
            config,
        )  # update parameters
        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % eval_interval == 0:
            policy_fn = partial(act_fn, train_state=train_state)
            normalized_score = evaluate(
                policy_fn,
                config.env_name,
                num_episodes=config.eval_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            print(i, normalized_score)
            eval_metrics = {f"{config.env_name}/episode_return": normalized_score}
            wandb.log(eval_metrics, step=i)
    
    # final evaluation
    policy_fn = partial(act_fn, train_state=train_state)
    normalized_score = evaluate(
        policy_fn,
        config.env_name,
        num_episodes=config.eval_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({f"{config.env_name}/final_episode_return": normalized_score})
    wandb.finish()