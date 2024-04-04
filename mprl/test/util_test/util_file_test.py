import numpy as np

import mprl.util as util


def test_join_path():
    paths = ["/", "tmp", "test", "test"]
    print(f"before: {paths}")
    joined = util.join_path(*paths)
    print(f"after: {joined}")


def test_mkdir():
    util.print_wrap_title("test_mkdir")
    paths = ["/", "tmp", "test", "test"]
    path = util.join_path(*paths)

    util.print_line_title("Expect False")
    print(f"dir exist? {util.remove_file_dir(path)}")
    util.mkdir(path)
    util.print_line_title("Expect True")
    print(f"dir exist? {util.remove_file_dir(path)}")


def test_remove_file_dir():
    util.print_wrap_title("test_remove_file_dir")
    paths = ["/", "tmp", "test", "test"]
    path = util.join_path(*paths)
    util.print_line_title("Expect False")
    print(f"dir exist before mkdir? {util.remove_file_dir(path)}")
    print("mkdir " + path)
    util.mkdir(path)
    util.print_line_title("Expect True")
    print(f"dir exist before removal? {util.remove_file_dir(path)}")
    util.print_line_title("Expect False")
    print(f"dir exist after removal? {util.remove_file_dir(path)}")


def test_dir_go_up():
    util.print_wrap_title("test_dir_go_up")
    print("current pwd: " + util.dir_go_up(num_level=0))
    print("2 up pwd: " + util.dir_go_up(num_level=2))


def test_get_dataset_dir():
    util.print_wrap_title("test_get_dataset_dir")
    print(util.get_dataset_dir("dummy"))


def test_get_media_dir():
    util.print_wrap_title("test_get_media_dir")
    print(util.get_media_dir("dummy"))


def test_get_config_type():
    util.print_wrap_title("test_get_config_type")
    print(util.get_config_type())


def test_get_config_path():
    util.print_wrap_title("test_get_config_path")
    config_type = util.get_config_type()
    for c_type in config_type:
        print(util.get_config_path("dummy", c_type))


def test_make_log_dir_with_time_stamp():
    util.print_wrap_title("test_get_log_dir")
    print(util.make_log_dir_with_time_stamp("dummy"))


def test_parse_config():
    util.print_wrap_title("test_parse_config")
    path = util.get_config_path("NMP_TEST_config")
    config_dict = util.parse_config(path)
    print(config_dict)


def test_dump_config():
    util.print_wrap_title("test_dump_config")
    util.dump_config({"a": 123, "b": 456}, "dummy", "/tmp")
    print(util.parse_config("/tmp/dummy.yaml", "test"))
    util.remove_file_dir("/tmp/dummy.yaml")


def test_dump_all_config():
    util.print_wrap_title("test_dump_all_config")
    configs = [{"a": 123, "b": 456}, {"c": 123, "d": 456}, {"e": 123, "f": 456}]
    util.dump_all_config(configs, "dummy2", "/tmp")
    print(util.parse_config("/tmp/dummy2.yaml", "test"))
    util.remove_file_dir("/tmp/dummy2.yaml")


def test_get_file_names_in_directory():
    util.print_wrap_title("test_get_file_names_in_directory")
    paths = ["/", "tmp"]
    path = util.join_path(*paths)
    print(util.get_file_names_in_directory(path))


def test_move_files_from_to():
    util.print_wrap_title("test_move_files_from_to")
    from_path = util.get_dataset_dir("dummy")
    to_path = util.join_path(*["/", "tmp"])
    util.move_files_from_to(from_dir=from_path, to_dir=to_path, copy=True)


def test_clean_and_get_tmp_dir():
    util.print_wrap_title("test_clean_and_get_tmp_dir")
    print(util.clean_and_get_tmp_dir())


def test_get_nn_save_paths():
    util.print_wrap_title("test_get_nn_save_paths")
    s_path, w_path = util.get_nn_save_paths("log", "test_nn", 1000)

    print(s_path)
    print(w_path)


def test_get_optimizer_save_path():
    util.print_wrap_title("test_get_optimizer_save_path")
    o_path = util.get_training_state_save_path("log", "opt", 1000)
    print(o_path)


def test_save_npz_dataset():
    util.print_wrap_title("test_save_npz_dataset")
    data_dict = {"a": np.random.randint(0, 10, [2, 2]),
                 "b": np.random.randint(0, 10, [2, 2])}
    util.save_npz_dataset("test_npz", overwrite=True, **data_dict)


def test_load_npz_dataset():
    util.print_wrap_title("test_load_npz_dataset")
    data_dict = util.load_npz_dataset("test_npz")
    print(data_dict)


def main():
    # test_join_path()
    # test_mkdir()
    # test_remove_file_dir()
    # test_dir_go_up()
    # test_get_dataset_dir()
    # test_get_media_dir()
    # test_get_config_type()
    # test_get_config_path()
    # test_make_log_dir_with_time_stamp()
    # test_parse_config()
    test_dump_config()
    test_dump_all_config()
    # test_get_file_names_in_directory()
    # test_move_files_from_to()
    # test_clean_and_get_tmp_dir()
    # test_get_nn_save_paths()
    # test_get_optimizer_save_path()
    # test_save_npz_dataset()
    # test_load_npz_dataset()


if __name__ == "__main__":
    main()
