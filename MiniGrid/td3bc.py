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


def generate_random_data(
    env_name: str, 
    num_episodes: int, 
    max_steps_per_episode: int = 100,
    seed: int = 42
) -> ProcessedTransition:
    """
    生成随机数据用于训练
    
    Args:
        env_name: 环境名称
        num_episodes: episode数量
        max_steps_per_episode: 每个episode的最大步数
        seed: 随机种子
        
    Returns:
        处理后的transition数据
    """
    print(f"Generating random data for {env_name}...")
    
    # 创建环境
    env, env_params = xminigrid.make(env_name)
    
    # 创建数据处理器
    processor = MinariDataProcessor(
        image_size=(22, 22),
        use_mission=False,
        normalize_image=True,
        normalize_direction=True
    )
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    all_dones = []
    
    rng = jax.random.PRNGKey(seed)
    
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Generating episode {episode}/{num_episodes}")
        
        # 重置环境
        rng, reset_rng = jax.random.split(rng)
        timestep = env.reset(env_params, reset_rng)
        
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        episode_next_observations = []
        episode_dones = []
        
        for step in range(max_steps_per_episode):
            # 处理observation - xminigrid的observation是直接的JAX数组
            obs_array = timestep.observation
            obs_numpy = np.array(obs_array)
            
            # xminigrid的observation形状是(7, 7, 2)
            # 第一个通道可能是物体类型，第二个通道可能是颜色
            # 将其转换为RGB图像格式
            if obs_numpy.shape == (7, 7, 2):
                # 将(7, 7, 2)转换为(7, 7, 3)的RGB图像
                # 这里使用简单的映射，实际可能需要更复杂的转换
                object_types = obs_numpy[:, :, 0]
                colors = obs_numpy[:, :, 1]
                
                # 创建RGB图像 (简化处理)
                rgb_image = np.zeros((7, 7, 3), dtype=np.uint8)
                rgb_image[:, :, 0] = colors  # R通道
                rgb_image[:, :, 1] = object_types  # G通道
                rgb_image[:, :, 2] = 0  # B通道
                
                # 上采样到22x22
                from scipy.ndimage import zoom
                try:
                    rgb_image = zoom(rgb_image, (22/7, 22/7, 1), order=0)
                except ImportError:
                    # 如果没有scipy，使用简单的重复
                    rgb_image = np.repeat(np.repeat(rgb_image, 3, axis=0), 3, axis=1)
                    rgb_image = rgb_image[:22, :22, :]
                
                direction = np.array([0.0])  # 默认方向
            else:
                # 其他形状的处理
                rgb_image = obs_numpy
                direction = np.array([0.0])
            
            obs_dict = {
                'image': rgb_image,
                'direction': direction
            }
            processed_obs = processor.process_observation(obs_dict)
            episode_observations.append(processed_obs)
            
            # 随机选择动作
            rng, action_rng = jax.random.split(rng)
            action = jax.random.randint(action_rng, (), 0, env.num_actions(env_params))
            episode_actions.append(action)
            
            # 执行动作
            rng, step_rng = jax.random.split(rng)
            next_timestep = env.step(env_params, timestep, action)
            
            episode_rewards.append(next_timestep.reward)
            episode_dones.append(next_timestep.is_done())
            
            # 处理next_observation
            if not next_timestep.is_done():
                next_obs_array = next_timestep.observation
                next_obs_numpy = np.array(next_obs_array)
                
                if next_obs_numpy.shape == (7, 7, 2):
                    # 同样的处理逻辑
                    next_object_types = next_obs_numpy[:, :, 0]
                    next_colors = next_obs_numpy[:, :, 1]
                    
                    next_rgb_image = np.zeros((7, 7, 3), dtype=np.uint8)
                    next_rgb_image[:, :, 0] = next_colors
                    next_rgb_image[:, :, 1] = next_object_types
                    next_rgb_image[:, :, 2] = 0
                    
                    try:
                        next_rgb_image = zoom(next_rgb_image, (22/7, 22/7, 1), order=0)
                    except ImportError:
                        next_rgb_image = np.repeat(np.repeat(next_rgb_image, 3, axis=0), 3, axis=1)
                        next_rgb_image = next_rgb_image[:22, :22, :]
                    
                    next_direction = np.array([0.0])
                else:
                    next_rgb_image = next_obs_numpy
                    next_direction = np.array([0.0])
                
                next_obs_dict = {
                    'image': next_rgb_image,
                    'direction': next_direction
                }
                processed_next_obs = processor.process_observation(next_obs_dict)
            else:
                # episode结束，next_observation设为0
                processed_next_obs = np.zeros_like(processed_obs)
            
            episode_next_observations.append(processed_next_obs)
            
            timestep = next_timestep
            
            if next_timestep.is_done():
                break
        
        # 将episode数据添加到总数据中
        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)
        all_next_observations.extend(episode_next_observations)
        all_dones.extend(episode_dones)
    
    # 转换为numpy数组
    all_observations = np.array(all_observations, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    all_next_observations = np.array(all_next_observations, dtype=np.float32)
    all_dones = np.array(all_dones, dtype=np.float32)
    
    print(f"Generated {len(all_observations)} transitions")
    
    return ProcessedTransition(
        observations=all_observations,
        actions=all_actions,
        rewards=all_rewards,
        next_observations=all_next_observations,
        dones=all_dones
    )


def get_dataset(config: TD3BCConfig) -> Tuple[Transition, np.ndarray, np.ndarray]:
    """
    获取数据集，支持从Minari加载或生成随机数据
    
    Args:
        config: 配置对象
        
    Returns:
        dataset, obs_mean, obs_std
    """
    if config.dataset_name is not None:
        # 从Minari加载数据
        print(f"Loading dataset from Minari: {config.dataset_name}")
        dataset, obs_mean, obs_std = load_and_process_minari_dataset(
            dataset_name=config.dataset_name,
            num_episodes=config.num_episodes,
            image_size=(22, 22),
            normalize=config.normalize_state
        )
    else:
        # 生成随机数据
        print("Generating random dataset...")
        dataset = generate_random_data(
            env_name=config.env_name,
            num_episodes=config.num_episodes,
            max_steps_per_episode=100,
            seed=config.seed
        )
        
        # 归一化observations
        obs_mean, obs_std = 0, 1
        if config.normalize_state:
            obs_mean = dataset.observations.mean(0)
            obs_std = dataset.observations.std(0) + 1e-5
            dataset = ProcessedTransition(
                observations=(dataset.observations - obs_mean) / obs_std,
                actions=dataset.actions,
                rewards=dataset.rewards,
                next_observations=(dataset.next_observations - obs_mean) / obs_std,
                dones=dataset.dones
            )
    
    # 限制数据大小
    data_size = min(config.data_size, len(dataset.observations))
    if len(dataset.observations) > data_size:
        rng = jax.random.PRNGKey(config.seed)
        rng, rng_permute, rng_select = jax.random.split(rng, 3)
        perm = jax.random.permutation(rng_permute, len(dataset.observations))
        dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
        dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    
    # 转换为Transition格式
    transition = Transition(
        observations=jnp.array(dataset.observations, dtype=jnp.float32),
        actions=jnp.array(dataset.actions, dtype=jnp.float32),
        rewards=jnp.array(dataset.rewards, dtype=jnp.float32),
        next_observations=jnp.array(dataset.next_observations, dtype=jnp.float32),
        dones=jnp.array(dataset.dones, dtype=jnp.float32),
    )
    
    return transition, obs_mean, obs_std


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