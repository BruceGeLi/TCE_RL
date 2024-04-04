import mprl.util as util


def test_get_git_repo_paths():
    util.print_wrap_title("test_get_git_repo_paths")

    repo_path_dict = util.get_git_repo_paths()
    for repo, path in repo_path_dict.items():
        print(f"{repo}: {path}")


def test_get_git_tracker():
    util.print_wrap_title("test_get_git_tracker")
    tracker = util.get_git_tracker()
    new_commit = tracker.get_git_repo_commits()
    print(new_commit)


def test_git_repos_old_vs_new():
    util.print_wrap_title("test_git_repos_old_vs_new")
    old_repo = {"mp_pytorch": "188c5734f4fb81c37fa00b2b31d11f468d77a774",
                "cw2": "15b52480fc6fac3e6f094dd29f4348c9eb163c50"}
    new_repos = {'cw2': '15b52480fc6fac3e6f094dd29f4348c9eb163c50',
                 'mp_pytorch': 'a69c3673439f3c5ce1e9e0cb945b31ad37a99a66',
                 'fancy_gym': '33bf76e9df28b27f5653571b5d74086ec52a128a',
                 'mprl': '0b55e978128296c0a11ea2d97f53385e7f8cf28a',
                 'trust_region_projections':
                     'ff7ea7b5d40b2d6bdca3d2a346fa7a51669f1cda'}
    diff = util.git_repos_old_vs_new(old_repo, new_repos)
    print(f"Git repositories commit check: ", diff, sep="\n")


def test_print_all_repo_branch_commit():
    util.print_wrap_title("test_print_all_repo_branch_commit")
    util.print_all_repo_branch_commit()


def main():
    test_get_git_repo_paths()
    test_get_git_tracker()
    test_git_repos_old_vs_new()
    test_print_all_repo_branch_commit()


if __name__ == "__main__":
    main()
