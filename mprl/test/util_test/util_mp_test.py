import mprl.util as util


def test_make_bb_vec_env(env_id):
    util.print_wrap_title("test_make_bb_vec_env")

    env = util.make_bb_vec_env(env_id, num_env=4, seed=0,
                         render=False,
                         mp_args={"num_basis": 8,
                                  "disable_goal": False})
    for i in range(5):
        obs = env.reset()
        print(f"{obs}")


def test_make_vec_env(env_id):
    util.print_wrap_title("test_make_bb_vec_env")
    util.make_bb_vec_env(env_id, num_env=1, seed=0, render=False, mp_args={})


def main():
    # util.set_logger_level("WARN")
    util.set_logger_level("ERROR")

    # test_make_bb_vec_env("metaworld/button-press-v2")
    # test_make_bb_vec_env("metaworld_ProDMP/button-press-v2")
    # test_make_bb_vec_env("metaworld_ProDMP_TCE/button-press-v2")
    # test_make_bb_vec_env("fancy_ProDMP/BoxPushingRandomInitDense-v0")
    # test_make_bb_vec_env("fancy_ProDMP_TCE/BoxPushingRandomInitDense-v0")
    # test_make_bb_vec_env("fancy_ProDMP/TableTennis4D-v0")
    test_make_bb_vec_env("fancy_ProDMP/TableTennisRndInit-v0")


if __name__ == "__main__":
    main()
