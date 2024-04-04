import os

from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from tqdm import tqdm

import mprl.util as util
from mprl.rl.agent import agent_factory
from mprl.rl.critic import critic_factory
from mprl.rl.policy import policy_factory
from mprl.rl.projection import projection_factory
from mprl.rl.sampler import sampler_factory


class MPExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int,
                   logger: cw_logging.LoggerArray) -> None:

        # Get experiment config
        cfg = cw_config["params"]
        cpu_cores = cw_config.get("cpu_cores", None)

        # Set random seed globally
        util.set_global_random_seed(cw_config["seed"])
        self.verbose_level = cw_config.get("verbose_level", 1)

        # Determine training or testing mode
        load_model_dir = cw_config.get('load_model_dir', None)
        load_model_epoch = cw_config.get('load_model_epoch', None)
        self.training = True if load_model_dir is None else False

        if self.training and cw_config.get("save_model_dir", None) is not None:
            # Save model in training mode
            self.save_model_dir = os.path.abspath(cw_config["save_model_dir"])
            self.save_model_interval = \
                max(cw_config["iterations"] // cw_config["num_checkpoints"], 1)

        else:
            # In testing mode or no save model dir in training mode
            self.save_model_dir = None
            self.save_model_interval = None

        # Components
        self.sampler = sampler_factory(cfg["sampler"]["type"],
                                       cpu_cores=cpu_cores,
                                       **cfg["sampler"]["args"])
        self.policy = policy_factory(cfg["policy"]["type"],
                                     dim_in=self.get_dim_in(cfg, self.sampler),
                                     dim_out=self.dim_policy_out(cfg),
                                     **cfg["policy"]["args"])
        self.critic = critic_factory(cfg["critic"]["type"],
                                     dim_in=self.get_dim_in(cfg, self.sampler),
                                     dim_out=1,
                                     **cfg["critic"]["args"])
        self.projection = projection_factory(cfg["projection"]["type"],
                                             action_dim=
                                             self.dim_policy_out(cfg),
                                             **cfg["projection"]["args"])
        self.agent = agent_factory(cfg["agent"]["type"],
                                   policy=self.policy,
                                   critic=self.critic,
                                   sampler=self.sampler,
                                   projection=self.projection,
                                   **cfg["agent"]["args"])

        # Load model if it in testing mode
        if self.training:
            util.print_line_title("Training")
        else:
            self.agent.load_agent(load_model_dir, load_model_epoch)
            util.print_line_title("Testing")

        # Progressbar
        self.progress_bar = tqdm(total=cw_config["iterations"])

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        if self.training:
            result_metrics = self.agent.step()

            self.progress_bar.update(1)
            if self.verbose_level == 0:
                return {}
            elif self.verbose_level == 1:
                for key in dict(result_metrics).keys():
                    if "exploration" in key:
                        del result_metrics[key]
                return result_metrics
            elif self.verbose_level == 2:
                return result_metrics

        else:
            # Note: Use the below line to train a loaded model for long term bugs
            # self.agent.step()

            deterministic_result_dict, _ = self.agent.evaluate(render=True)
            self.progress_bar.update(1)
            return deterministic_result_dict

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0
                                    or (n + 1) == cw_config["iterations"]):
            self.agent.save_agent(log_dir=self.save_model_dir, epoch=n + 1)

    def finalize(self,
                 surrender: cw_error.ExperimentSurrender = None,
                 crash: bool = False):
        pass

    @staticmethod
    def get_dim_in(cfg, sampler):
        """
        Get the dimension of the policy and critic input

        Args:
            cfg: config dict
            sampler: sampler of the experiment

        Returns:
            dim_in: dimension of the policy output

        """
        if "TemporalCorrelated" in cfg["sampler"]["type"]:
            dof = cfg["mp"]["args"]["num_dof"]
            return sampler.observation_shape[-1] - dof * 2
        else:
            return sampler.observation_shape[-1]

    @staticmethod
    def dim_policy_out(cfg):
        """
        Get the dimension of the policy output

        Args:
            cfg: config dict

        Returns:
            dim_out: dimension of the policy output

        """
        mp_type = cfg["mp"]["type"]
        dof = cfg["mp"]["args"]["num_dof"]
        num_basis = cfg["mp"]["args"]["num_basis"]
        learn_tau = cfg["mp"]["args"].get("learn_tau", False)
        learn_delay = cfg["mp"]["args"].get("learn_delay", False)

        if mp_type == "prodmp":
            dim_out = dof * (num_basis + 1)  # weights + goal

            # Disable goal if specified
            if cfg["mp"]["args"].get("disable_goal", False):
                dim_out -= dof

        elif mp_type == "promp":
            dim_out = dof * num_basis  # weights only
        else:
            raise NotImplementedError

        if learn_tau:
            dim_out += 1
        if learn_delay:
            dim_out += 1

        return dim_out


def evaluation(model_str: str, version_number: list, epoch: int,
               keep_training: bool):
    """
    Given wandb model string, version, and epoch number, evaluate the model
    Args:
        model_str: wandb model string
        version_number: number of the version
        epoch: epoch number of the model
        keep_training: whether to keep training the model

    Returns:
        None
    """
    for v_num in version_number:
        util.RLExperiment(MPExperiment, False, model_str, v_num, epoch,
                          keep_training)


if __name__ == "__main__":
    for key in os.environ.keys():
        if "-xCORE-AVX2" in os.environ[key]:
            os.environ[key] = os.environ[key].replace("-xCORE-AVX2", "")

    util.RLExperiment(MPExperiment, True)
