from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Non-ctx-cov
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "box_random_bbrl_entire/model:version', type='model')"

    #================================================
    version_number = [32]
    epoch = 6000

    evaluation(model_str, version_number, epoch, False)
