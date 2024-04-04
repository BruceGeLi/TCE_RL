"""
    Utilities of check local git repositories status
"""
import inspect

import git_repos_tracker.tracker

import mprl.util as util


def get_git_repo_paths() -> dict:
    """
    Get pre-registered modules' repository path

    Returns:
        path to the repositories

    """
    module_path_dict = dict()
    repo_path_dict = dict()

    import mprl
    module_path_dict["mprl"] = inspect.getfile(mprl)
    # import mp_pytorch
    # module_path_dict["mp_pytorch"] = inspect.getfile(mp_pytorch)
    import fancy_gym
    module_path_dict["fancy_gym"] = inspect.getfile(fancy_gym)
    import metaworld
    module_path_dict["metaworld"] = inspect.getfile(metaworld)
    import trust_region_projections
    module_path_dict["trust_region_projections"] = \
        inspect.getfile(trust_region_projections)
    # import cw2
    # module_path_dict["cw2"] = inspect.getfile(cw2)

    import git_repos_tracker
    module_path_dict["git_repos_tracker"] = inspect.getfile(git_repos_tracker)

    for module_name, path in module_path_dict.items():
        if "site-packages" in path:
            raise RuntimeError(f"Module {module_name} is in site_packages, "
                               f"thus cannot track its repository.")
        repo_path_dict[module_name] = util.dir_go_up(num_level=2,
                                                     current_file_dir=path)

    return repo_path_dict


def git_repos_old_vs_new(old: dict, new: dict):
    new_added_repo = []
    mismatch_repo = []

    for repo_name in new.keys():
        if repo_name not in old.keys():
            new_added_repo.append(repo_name)
            continue
        if new[repo_name] != old[repo_name]:
            mismatch_repo.append({f"{repo_name}_old": old[repo_name],
                                  f"{repo_name}_now": new[repo_name]})

    result = dict()
    if len(new_added_repo) > 0:
        result["new_added_repo"] = new_added_repo
    if len(mismatch_repo) > 0:
        result["mismatch_repo"] = mismatch_repo

    return result


def get_git_tracker():
    """
    Return the git repositories tracker for current project

    Returns:
        tracker of repositories
    """
    return git_repos_tracker.tracker.GitReposTracker(get_git_repo_paths)


def print_all_repo_branch_commit():
    git_tracker = get_git_tracker()
    git_tracker.check_clean_git_status(print_result=True)