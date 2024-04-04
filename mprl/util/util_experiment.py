import copy
import os.path
import sys

from typing import Union

import torch
import wandb
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger
from cw2.experiment import AbstractExperiment
from cw2.experiment import AbstractIterativeExperiment

import mprl.util as util


def is_on_local_machine():
    if any(["local" in argv for argv in sys.argv]):
        return True
    else:
        return False


def is_slurm(cw: cluster_work.ClusterWork):
    if cw.args["slurm"]:
        return True
    else:
        return False


def download_saved_model(model_str: str, model_version: int):
    model_api = model_str.replace("version", f"v{model_version}")
    run = wandb.init()
    artifact = eval(model_api[11:])
    download_dir = util.make_log_dir_with_time_stamp("/tmp/download_model")
    artifact.download(root=download_dir)
    file_names = util.get_file_names_in_directory(download_dir)
    file_names.sort()
    util.print_line_title(title=f"Download model {model_version} from WandB")
    path_to_old_config = f"{download_dir}/config.yaml"
    return download_dir, path_to_old_config


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def assign_env_to_cpu(num_env, envs, cpu_ids):
    if cpu_ids is not None and num_env > 1:
        assert len(cpu_ids) >= num_env, ("The number of cpu cores should be >= "
                                         "the num of environments.")
        cores_per_env = int(len(cpu_ids) / num_env)
        cpu_cores_list = list(cpu_ids)
        env_pids = [envs.processes[i].pid for i in range(num_env)]
        for i, pid in enumerate(env_pids):
            cores_env = cpu_cores_list[i * cores_per_env:
                                       (i + 1) * cores_per_env]
            assign_process_to_cpu(pid, set(cores_env))
    else:
        pass


def set_logger_level(level: str):
    """
    Set the logger level
    Args:
        level: "DEBUG", "INFO", "WARN", "ERROR"

    Returns:

    """
    assert level in ["DEBUG", "INFO", "WARN", "ERROR"]
    from gymnasium import logger
    eval("logger.setLevel(logger." + level + ")")


class RLExperiment:
    def __init__(self, exp: Union[AbstractExperiment,
    AbstractIterativeExperiment],
                 train: bool,
                 model_str: str = None,
                 version_number: int = None,
                 epoch: int = None,
                 keep_training: bool = False):

        git_tracker = util.get_git_tracker()

        # Determine if check git status
        if util.is_debugging():
            util.print_line_title('Debug mode, do not check git repo commits.')
            # Use TkAgg backend for matplotlib to avoid error against mujoco_py
            import matplotlib
            matplotlib.use('TkAgg')
        else:
            util.print_line_title('Run mode, enforce git repo commit checking.')
            git_clean, git_status = git_tracker.check_clean_git_status(
                print_result=True)
            if not git_clean:
                assert False, "Repositories not clean"

        self.current_git_commits = git_tracker.get_git_repo_commits()

        if train:
            # Initialize experiment
            self.cw = cluster_work.ClusterWork(exp)

            # Process configs if running on local machine or slurm on cluster
            if is_slurm(self.cw) or is_on_local_machine():
                self._process_train_rep_config_file(self.cw.config)

        else:
            # Download the saved model and add to system arguments
            download_dir, path_to_old_config \
                = download_saved_model(model_str, version_number)
            sys.argv.extend([path_to_old_config, "-o", "--nocodecopy"])

            # Compare the current and the old commits
            self.old_git_commits = \
                util.parse_config(path_to_old_config)[0]["git_repos"]
            util.print_wrap_title("Git repos commits check")
            print(util.git_repos_old_vs_new(self.old_git_commits,
                                            self.current_git_commits))

            # Initialize experiment
            self.cw = cluster_work.ClusterWork(exp)

            # Process configs
            self._process_test_rep_config_file(self.cw.config.exp_configs,
                                               download_dir, epoch,
                                               keep_training)

        # Add wandb logger
        if not util.is_debugging():
            self.cw.add_logger(WandBLogger())
        self.cw.run()

    def _process_train_rep_config_file(self, config_obj):
        """
        Given processed cw2 configuration, do further process, including:
        - Overwrite log path with time stamp
        - Create model save folders
        - Overwrite random seed by the repetition number
        - Save the current repository commits
        - Make a copy of the config and restore the exp path to the original
        - Dump this copied config into yaml file into the model save folder
        - Dump the current time stamped config file in log folder to make slurm
          call bug free
        Args:
            exp_configs: list of configs processed by cw2 already

        Returns:
            None

        """
        exp_configs = config_obj.exp_configs
        formatted_time = util.get_formatted_date_time()
        # Loop over the config of each repetition
        for i, rep_config in enumerate(exp_configs):

            # Add time stamp to log directory
            log_path = rep_config["log_path"]
            rep_log_path = rep_config["_rep_log_path"]
            rep_config["log_path"] = \
                log_path.replace("log", f"log_{formatted_time}")
            rep_config["_rep_log_path"] = \
                rep_log_path.replace("log", f"log_{formatted_time}")

            # Make model save directory
            model_save_dir = util.join_path(rep_config["_rep_log_path"],
                                            "model")
            try:
                util.mkdir(os.path.abspath(model_save_dir))
            except FileExistsError:
                import logging
                logging.error(formatted_time)
                raise

            # Set random seed to the repetition number
            util.set_value_in_nest_dict(rep_config, "seed",
                                        rep_config['_rep_idx'])

            # Save repo commits
            rep_config["git_repos"] = self.current_git_commits

            # Make a hard copy of the config
            copied_rep_config = copy.deepcopy(rep_config)

            # Recover the path to its original
            copied_rep_config["path"] = copied_rep_config["_basic_path"]

            # Reset the repetition number to 1 for future test usage
            copied_rep_config["repetitions"] = 1
            if copied_rep_config.get("reps_in_parallel", False):
                del copied_rep_config["reps_in_parallel"]
            if copied_rep_config.get("reps_per_job", False):
                del copied_rep_config["reps_per_job"]

            # Delete the generated cw2 configs
            for key in rep_config.keys():
                if key[0] == "_":
                    del copied_rep_config[key]
            del copied_rep_config["log_path"]

            # Save this copied subconfig file
            util.dump_config(copied_rep_config, "config",
                             os.path.abspath(model_save_dir))

        # Save the time stamped config file in local /log directory
        time_stamped_config_path = util.make_log_dir_with_time_stamp("")
        util.mkdir(time_stamped_config_path, overwrite=True)

        config_obj.to_yaml(time_stamped_config_path,
                           relpath=False)
        config_obj.config_path = \
            util.join_path(time_stamped_config_path,
                           "relative_" + config_obj.f_name)

    @staticmethod
    def _process_test_rep_config_file(exp_configs, load_model_dir, epoch,
                                      keep_training):
        """
        Given processed cw2 configuration, do further process, including:
        - Overwrite log path with time stamp
        - Create model save folders
        - Overwrite random seed by the repetition number
        - Save the current repository commits
        - Make a copy of the config and restore the exp path to the original
        - Dump this copied config into yaml file into the model save folder

        Args:
            exp_configs: list of configs processed by cw2 already
            load_model_dir: model saved dir
            epoch: epoch of the model
            keep_training: whether to keep training
        Returns:
            None

        """
        assert len(exp_configs) == 1
        formatted_time = util.get_formatted_date_time()
        test_config = exp_configs[0]

        # Add time stamp to log directory
        log_path = test_config["log_path"]
        rep_log_path = test_config["_rep_log_path"]
        test_config["log_path"] = \
            log_path.replace("log", f"log_{formatted_time}")
        test_config["_rep_log_path"] = \
            rep_log_path.replace("log", f"log_{formatted_time}")
        test_config["load_model_dir"] = load_model_dir
        test_config["load_model_epoch"] = epoch
        test_config["repetitions"] = 1
        test_config["reps_in_parallel"] = 1
        test_config["reps_per_job"] = 1
        test_config["params"]["sampler"]["args"]["num_env_test"] = 1
        if not keep_training:
            test_config["params"]["sampler"]["args"]["num_env_train"] = 1


def make_mdp_reward(task_id: str, step_rewards, step_infos,
                    dtype, device):
    """
    This is a helper function to turn the non-MDP reward into MDP reward, by
    summing up the non-MDP reward after the event happens and set the summed
    reward at the event index.

    The event includes, e.g.: table tennis ball hit, hopper jump off the floor.

    Why do we need this? Because the non-MDP reward is not compatible with the
    RL architecture with MDP assumption. As the V(s) cannot fully describe the
    expected return of the state, because the reward is determined by the
    actions happened in the past, rather than the current state.

    E.g. in the table tennis task, the reward after hitting is not determined by
    the movement of the robot at the moment. So the current state of the task
    cannot be used for the prediction of the future reward.

    This function detects then key event and sum up the reward after the event
    Then set the summed reward at the event index.

    Hit:            False, False, False, False, True, True, True, True, True
    Non-MDP Reward:     1,     1,     1,     1,    2,    2,    2,    2,    2
    MDP Reward:         1,     1,     1,     1,   10,    0,    0,    0,    0

    If the event never happens, the MDP reward is the same as the non-MDP reward

    """

    if "TableTennis" in task_id:
        mdp_key_info = "hit_ball"
    elif "HopperJump" in task_id:
        mdp_key_info = "has_left_floor"
    else:
        return step_rewards

    # Get the index when (and after) the event happened
    event_info = util.get_item_from_dicts(step_infos, mdp_key_info)
    event_index = torch.where(torch.as_tensor(event_info, device=device), 1.0,
                              0.0)

    # Get the exact index of the event
    event_index_first = torch.argmax(event_index, dim=-1)

    # Sum up the reward after the event
    reward_after_event = step_rewards * event_index
    reward_after_event = reward_after_event.sum(axis=-1)

    # Check if the event happened
    event_happened = event_index_first > 0

    # Set the summed reward at the event index
    step_rewards[event_happened, event_index_first[event_happened]] \
        = reward_after_event[event_happened]

    # Get the mask for the index after the event happened
    mask_after_event = (torch.arange(step_rewards.size(1),
                                     device=device).unsqueeze(
        0) > event_index_first.unsqueeze(1))
    mask_combined = torch.logical_and(event_happened.unsqueeze(1),
                                      mask_after_event)

    # Set the reward after the event to 0
    step_rewards[mask_combined] = 0

    return step_rewards
