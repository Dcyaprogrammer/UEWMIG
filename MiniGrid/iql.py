import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax
import flax.linen as nn
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
import timeit
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
import xminigrid
import distrax
from xminigrid.wrappers import GymAutoResetWrapper
import os
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple



# Collect Rollouts
def build_rollout(env, env_params, num_steps):
  def rollout(rng):
    def _step_fn(carry, _):
      rng, timestep = carry
      rng, _rng = jax.random.split(rng)
      action = jax.random.randint(_rng, shape=(), minval=0, maxval=env.num_actions(env_params))

      timestep = env.step(env_params, timestep, action)

      return (rng, timestep), (timestep,action)

    rng, _rng = jax.random.split(rng)
    timestep = env.reset(env_params, _rng)
    rng, (transitions, actions) = jax.lax.scan(_step_fn, (rng, timestep), None, length=num_steps)

    return transitions, actions
  return rollout


# Build Environment
env, env_params = xminigrid.make("MiniGrid-EmptyRandom-6x6")
env = GymAutoResetWrapper(env)

rollout_fn = jax.jit(build_rollout(env, env_params, num_steps=1000))

transitions, actions = rollout_fn(jax.random.key(0))



def create_replay_buffer(transitions, actions):

  observations = transitions.observation # (T, 7, 7, 2)
  rewards = transitions.reward # (T,)
  dones = transitions.step_type == 2 # (T,)
  next_observations = jnp.concatenate([observations[1:], observations[-1:]], axis=0) #(T, 7, 7, 2)
  actions = jnp.array(actions, dtype=jnp.int32) #(T,)

  replay_buffer = {'observations': observations,
                   'actions': actions,
                   'rewards': rewards,
                   'next_observations': next_observations,
                   'dones': dones}

  # print("=== Replay Buffer 构建完成 ===")
  # print(f"数据点数量: {len(observations)}")
  # print(f"平均奖励: {jnp.mean(rewards):.4f}")
  # print(f"Episode结束次数: {jnp.sum(dones)}")
  # print(f"动作分布: {jnp.bincount(actions)}")
  return replay_buffer

replay_buffer = create_replay_buffer(transitions, actions)

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

class IQLConfig(BaseModel):
    # GENERAL
    algo: str = "IQL"
    project: str = "train-IQL"
    env_name: str = "MiniGrid-EmptyRandom-6x6"
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 100000
    batch_size: int = 256
    max_steps: int = int(1e3)
    n_jitted_updates: int = 8
    # DATASET
    data_size: int = int(1e6)
    normalize_state: bool = False
    normalize_reward: bool = True
    # NETWORK
    hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    layer_norm: bool = True
    opt_decay_schedule: bool = True
    # IQL SPECIFIC
    expectile: float = (
        0.7  # FYI: for Hopper-me, 0.5 produce better result. (antmaze: expectile=0.9)
    )
    beta: float = (
        3.0  # FYI: for Hopper-me, 6.0 produce better result. (antmaze: beta=10.0)
    )
    tau: float = 0.005
    discount: float = 0.99

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = IQLConfig(**conf_dict)

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
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        batch_size = observations.shape[0]
        actions = jax.nn.one_hot(actions, num_classes=4) #one-hot encoding
        flat_observations = observations.reshape(batch_size, -1)
        inputs = jnp.concatenate([flat_observations, actions], axis=-1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        batch_size = observations.shape[0]
        obs_flat = observations.reshape(batch_size, -1)
        critic = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm)(obs_flat)
        return jnp.squeeze(critic, -1)


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init()
        )(outputs)
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution
  
class CatPolicy(nn.Module):
  hidden_dims : Sequence[int]
  action_dim: int

  @nn.compact
  def __call__(self, observations: jnp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
    x = observations.reshape(observations.shape[0], -1) # flatten
    outputs = MLP(self.hidden_dims, activate_final=True)(x)
    logits = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
    logits = logits / temperature
    distribution = distrax.Categorical(logits=logits)
    return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    dones_float: jnp.ndarray

def get_normalization(dataset: Transition) -> float:
    # into numpy.ndarray
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000

def preprocess_dataset(
     dataset: dict, config: IQLConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = jnp.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset['dones'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-6 or dataset['dones'][i] == True:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dataset["dones"], dtype=jnp.float32),
        dones_float=jnp.array(dones_float, dtype=jnp.float32),
    )

    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    # normalize rewards
    if config.normalize_reward:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)

    # shuffle data and select the first data_size samples
    data_size = min(config.data_size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    return dataset, obs_mean, obs_std

def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

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

class IQLTrainState(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState

class IQL(object):

    @classmethod
    def update_critic(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        next_v = train_state.value.apply_fn(
            train_state.value.params, batch.next_observations
        )
        target_q = batch.rewards + config.discount * (1 - batch.dones) * next_v

        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q1, q2 = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(
            train_state.critic, critic_loss_fn
        )
        return train_state._replace(critic=new_critic), critic_loss

    @classmethod
    def update_value(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        q1, q2 = train_state.target_critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
        def value_loss_fn(value_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = train_state.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.expectile).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.value, value_loss_fn)
        return train_state._replace(value=new_value), value_loss

    @classmethod
    def update_actor(
        self, train_state: IQLTrainState, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainState", Dict]:
        v = train_state.value.apply_fn(train_state.value.params, batch.observations)
        q1, q2 = train_state.critic.apply_fn(
            train_state.target_critic.params, batch.observations, batch.actions
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * config.beta)
        exp_a = jnp.minimum(exp_a, 100.0)
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions.astype(jnp.int32))
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @classmethod
    def update_n_times(
        self,
        train_state: IQLTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: IQLConfig,
    ) -> Tuple["IQLTrainState", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, value_loss = self.update_value(train_state, batch, config)
            train_state, actor_loss = self.update_actor(train_state, batch, config)
            train_state, critic_loss = self.update_critic(train_state, batch, config)
            new_target_critic = target_update(
                train_state.critic, train_state.target_critic, config.tau
            )
            train_state = train_state._replace(target_critic=new_target_critic)
        return train_state, {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    @classmethod
    def get_action(
        self,
        train_state: IQLTrainState,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:

        # modified for discrete actions
        dist = train_state.actor.apply_fn(
            train_state.actor.params, observations, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        return actions
    
def create_iql_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: IQLConfig,
) -> IQLTrainState:
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = 4

    # Gaussian Model
    # actor_model = GaussianPolicy(
    #     config.hidden_dims,
    #     action_dim=action_dim,
    #     log_std_min=-5.0,
    # )

    # Cat Model
    actor_model = CatPolicy(
        config.hidden_dims,
        action_dim = action_dim
    )
    
    if config.opt_decay_schedule:
        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_steps)
        actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    else:
        actor_tx = optax.adam(learning_rate=config.actor_lr)
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=actor_tx,
    )
    # initialize critic
    critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # initialize value
    value_model = ValueCritic(config.hidden_dims, layer_norm=config.layer_norm)
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(value_rng, observations),
        tx=optax.adam(learning_rate=config.value_lr),
    )
    return IQLTrainState(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
    )

def evaluate(
    policy_fn, env, env_params, num_episodes: int, obs_mean: float, obs_std: float, rng
) -> float:
    print("evaluation started")
    episode_returns = []

    for i in range(num_episodes):
      print(f"episode {i} started")
      rng, _rng = jax.random.split(rng)
      episode_return = 0

      timestep = env.reset(env_params, _rng)
      done = timestep.step_type == 2
      observation = timestep.observation

      while not done:
          # potential case issue
          obs = observation[None, ...]
          action = policy_fn(observations=obs)

          if isinstance(action, (jnp.ndarray, np.ndarray)) and action.shape == (1,):
            action = int(action[0])
            
          timestep = env.step(env_params, timestep, action)
          reward = timestep.reward
          done = timestep.step_type == 2
          print(done)
          observation = timestep.observation

          episode_return += reward
      episode_returns.append(episode_return)
    return float(jnp.mean(jnp.array(episode_returns)))

if __name__ == "__main__":
    # wandb.init(config=config, project=config.project)

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    env, env_params = xminigrid.make("MiniGrid-EmptyRandom-6x6")
    env = GymAutoResetWrapper(env)

    dataset, obs_mean, obs_std = preprocess_dataset(replay_buffer, config)

    # create train_state
    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state: IQLTrainState = create_iql_train_state(
        _rng,
        example_batch.observations[None, ...],
        example_batch.actions[None, ...],
        config,
    )

    algo = IQL()
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,))
    act_fn = jax.jit(algo.get_action)
    num_steps = config.max_steps // config.n_jitted_updates
    eval_interval = config.eval_interval // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        train_state, update_info = update_fn(train_state, dataset, subkey, config)

        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            # wandb.log(train_metrics, step=i)

        # if i % eval_interval == 0:
        #     policy_fn = partial(
        #         act_fn,
        #         temperature=0.0,
        #         seed=jax.random.PRNGKey(0),
        #         train_state=train_state,
        #     )
        #     normalized_score = evaluate(
        #         policy_fn,
        #         env,
        #         num_episodes=config.eval_episodes,
        #         obs_mean=obs_mean,
        #         obs_std=obs_std,
        #     )
        #     print(i, normalized_score)
        #     eval_metrics = {f"{config.env_name}/normalized_score": normalized_score}
        #     wandb.log(eval_metrics, step=i)
    # final evaluation
    # policy_fn = partial(
    #     act_fn,
    #     temperature=0.0,
    #     seed=jax.random.PRNGKey(0),
    #     train_state=train_state,
    # )
    # normalized_score = evaluate(
    #     policy_fn,
    #     env,
    #     num_episodes=config.eval_episodes,
    #     obs_mean=obs_mean,
    #     obs_std=obs_std,
    # )
    # print("Final Evaluation", normalized_score)
    # wandb.log({f"{config.env_name}/final_normalized_score": normalized_score})
    # wandb.finish()
