from gym.spaces import Dict

import torch
import torch.nn as nn

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN

from gym_wordle.envs.wordle_rl import WordleEnv


class ActionMaskModel(TorchModelV2, nn.Module):
    """
    Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    PyTorch version of ActionMaskModel, derived from:
    https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
    https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        # self.register_variables(self.internal_model.variables())


    def forward(self, input_dict, state, seq_lens):
        # Extract available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model(
            {"obs": input_dict["obs"]["observations"]}
        )

        # Convert action_mask into a [0.0 || -inf] mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state


    def value_function(self):
        return self.internal_model.value_function()


if __name__ == "__main__":

    env_config = {
        'mask': True,
    }
    register_env("wordle-v0", lambda config: WordleEnv(config=env_config))

    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    # breakpoint()
    config = ppo.DEFAULT_CONFIG.copy()
    # config = {
    #     'use_critic': True,
    #     'use_gae': True,
    #     'lambda': 1.0,
    #     'kl_coeff': 0.2,
    #     'sgd_minibatch_size': 128,
    #     'shuffle_sequences': True,
    #     'num_sgd_iter': 30,
    #     'lr': 5e-5,
    #     'vf_loss_coeff': 1.0,
    #     'entropy_coeff': 0.0,
    #     'clip_param': 0.3,
    #     'vf_clip_param': 10.0,
    #     'kl_target': 0.01,
    #     'batch_mode': "complete_episodes",
    #     'log_sys_usage': True,
    #     'output_max_file_size': 67108864,
    #     'output_compress_columns': ['obs', 'new_obs'],
    #     'input_evaluation': ['is', 'wis'],
    #     'num_gpus': 1,
    #     'log_level': "WARN",
    #     'framework': "torch",
    #     'explore': True,
    #     'exploration_config': {'type': 'StochasticSampling'},
    #     'callbacks': ray.rllib.agents.callbacks.DefaultCallbacks
    # }
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["num_gpus"] = 1
    # config["batch_mode"] = "complete_episodes"
    config["num_gpus_per_worker"] = 0.25
    config["sgd_minibatch_size"] = 32
    # config["ignore_worker_failures"] = True
    config["model"] = {
        "custom_model": "action_mask_model",
    }


    ### instantiate the model with these
    # obs_space,
    # action_space,
    # num_outputs,
    # model_config,
    # name,

    # agent = ActionMaskModel(action_space, num_outputs, config, name, env="wordle-v0")
    agent = ppo.PPOTrainer(config, env="wordle-v0")

    n_episodes = 100

    for n in range(n_episodes):
        result = agent.train()
        chkpt_file = agent.save("checkpoints")
        print(
            f"Episode {n}: reward {result['episode_reward_mean']} in {result['episode_len_mean']} steps, saved {chkpt_file}"
        )
