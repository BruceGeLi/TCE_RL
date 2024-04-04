"""
    Utilities of files operation
"""

import os
import shutil
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import yaml
from natsort import os_sorted

import mprl.util as util
import logging

def join_path(*paths: Union[str]) -> str:
    """

    Args:
        *paths: paths to join

    Returns:
        joined path
    """
    return os.path.join(*paths)


def mkdir(directory: str, overwrite: bool = False):
    """

    Args:
        directory: dir path to make
        overwrite: overwrite exist dir

    Returns:
        None

    Raise:
        FileExistsError if dir exists and overwrite is False
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=overwrite)
    except FileExistsError:
        logging.error("Directory already exists, remove it before make a new one.")
        raise


def remove_file_dir(path: str) -> bool:
    """
    Remove file or directory
    Args:
        path: path to directory or file

    Returns:
        True if successfully remove file or directory

    """
    if not os.path.exists(path):
        return False
    elif os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
        return True
    else:
        shutil.rmtree(path)
        return True


def dir_go_up(num_level: int = 2, current_file_dir: str = "default") -> str:
    """
    Go to upper n level of current file directory
    Args:
        num_level: number of level to go up
        current_file_dir: current dir

    Returns:
        dir n level up
    """
    if current_file_dir == "default":
        current_file_dir = os.path.realpath(__file__)
    while num_level != 0:
        current_file_dir = os.path.dirname(current_file_dir)
        num_level -= 1
    return current_file_dir


def get_dataset_dir(dataset_name: str) -> str:
    """
    Get the path to the directory storing the dataset
    Args:
        dataset_name: name of the dataset

    Returns:
        path to the directory storing the dataset
    """
    return os.path.join(dir_go_up(2), "dataset", dataset_name)


def get_media_dir(media_name: str) -> str:
    """
    Get the path to the directory storing the media
    Args:
        media_name: name of the media

    Returns:
        path to the directory storing the media_name
    """
    return os.path.join(dir_go_up(2), "media", media_name)


def get_config_type() -> set:
    """
    Register current config type
    Returns:

    """
    return {"test",
            "reacher5d",
            "box_push_dense"}


def get_config_path(config_name: str, config_type: str = "test") -> str:
    """
    Get the path to the config file
    Args:
        config_name: name of the config file
        config_type: configuration type

    Returns:
        path to the config file
    """
    # Check config type
    assert config_type in get_config_type(), \
        "Unknown config type."
    return os.path.join(dir_go_up(2), "config", config_type,
                        config_name + ".yaml")


def make_log_dir_with_time_stamp(log_name: str) -> str:
    """
    Get the dir to the log
    Args:
        log_name: log's name

    Returns:
        directory to log file
    """

    return os.path.join(dir_go_up(2), "log", log_name,
                        util.get_formatted_date_time())


def parse_config(config_path: str, config_type: str = "test") \
        -> Union[dict, list]:
    """
    Parse config file into a dictionary
    Args:
        config_path: path to config file
        config_type: configuration type

    Returns:
        configuration in dictionary
    """
    assert config_type in get_config_type(), \
        "Unknown config type"

    all_config = list()
    with open(config_path, "r") as f:
        for config in yaml.load_all(f, yaml.FullLoader):
            all_config.append(config)
    return all_config


def dump_config(config_dict: dict, config_name: str, dump_dir: str):
    """
    Dump configuration into yaml file
    Args:
        config_dict: config dictionary to be dumped
        config_name: config file name
        dump_dir: dir to dump
    Returns:
        None
    """

    # Generate config path
    dump_path = util.join_path(dump_dir, config_name + ".yaml")

    # Remove old config if exists
    remove_file_dir(dump_path)

    # Write new config to file
    with open(dump_path, "w") as f:
        yaml.dump(config_dict, f)


def dump_all_config(list_config_dict: List[Dict], config_name: str,
                    dump_dir: str):
    """
    Dump multi configuration into yaml file
    Args:
        list_config_dict: list of config dictionaries to be dumped
        config_name: config file name
        dump_dir: dir to dump
    Returns:
        None
    """
    # Generate config path
    dump_path = util.join_path(dump_dir, config_name + ".yaml")

    # Remove old config if exists
    remove_file_dir(dump_path)

    # Write new config to file
    with open(dump_path, "w") as f:
        yaml.dump_all(list_config_dict, f)


def get_file_names_in_directory(directory: str) -> [str]:
    """
    Get file names in given directory
    Args:
        directory: directory where you want to explore

    Returns:
        file names in a list

    """
    file_names = None
    try:
        (_, _, file_names) = next(os.walk(directory))
        if len(file_names) == 0:
            file_names = None
    except StopIteration as e:
        print("Cannot read files from directory: ", directory)
        raise StopIteration("Cannot read files from directory")
    return os_sorted(file_names)


def move_files_from_to(from_dir: str,
                       to_dir: str,
                       copy=False):
    """
    Move or copy files from one directory to another
    Args:
        from_dir: from directory A
        to_dir: to directory B
        copy: True if copy instead of move

    Returns:
        None
    """
    file_names = get_file_names_in_directory(from_dir)
    for file in file_names:
        from_path = os.path.join(from_dir, file)
        to_path = os.path.join(to_dir, file)
        if copy:
            shutil.copy(from_path, to_path)
        else:
            shutil.move(from_path, to_path)


def clean_and_get_tmp_dir() -> str:
    """
    Get the path to the tmp folder

    Returns:
        path to the tmp directory
    """
    tmp_path = os.path.join(dir_go_up(2), "tmp")
    remove_file_dir(tmp_path)
    util.mkdir(tmp_path)
    return tmp_path


def get_nn_save_paths(log_dir: str, nn_name: str,
                      epoch: Optional[int]) -> Tuple[str, str]:
    """
    Get path storing nn structure parameters and nn weights
    Args:
        log_dir: directory to log
        nn_name: name of NN
        epoch: number of training epoch

    Returns:
        path to nn structure parameters
        path to nn weights
    """
    s_path = os.path.join(log_dir, nn_name + "_parameters.pkl")
    w_path = os.path.join(log_dir, nn_name + "_weights")
    if epoch is not None:
        w_path = w_path + "_{:d}".format(epoch)

    return s_path, w_path


def get_training_state_save_path(log_dir: str, name: str,
                                 epoch: Optional[int]) -> str:
    """
    Get path storing a training state, e.g. optimizer

    Args:
        log_dir: directory to save
        name: name of the optimizer
        epoch: number of training epoch

    Returns:
        path to save optimizer
    """
    o_path = os.path.join(log_dir, name + "_state")
    if epoch is not None:
        o_path = o_path + "_{:d}".format(epoch)
    return o_path


def save_npz_dataset(dataset_name: str, name: str = None,
                     overwrite: bool = False, **data_dict):
    if name is None:
        name = dataset_name
    save_dir = get_dataset_dir(dataset_name)
    mkdir(save_dir, overwrite=overwrite)
    np.savez(join_path(save_dir, name + ".npz"),
             **data_dict)


def load_npz_dataset(dataset_name: str, name: str = None) -> dict:
    if name is None:
        name = dataset_name
    load_dir = get_dataset_dir(dataset_name)
    load_path = join_path(load_dir, name + ".npz")
    data_dict = dict(np.load(load_path, allow_pickle=True))

    for key, value in data_dict.items():
        if value.shape == ():
            data_dict[key] = value.item()

    return data_dict
