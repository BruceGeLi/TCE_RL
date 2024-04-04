from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Non-ctx-cov
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "metaworld_bbrl_entire/model:version', type='model')"

    #================================================
    version_number = [1979]
    epoch = 2500

    evaluation(model_str, version_number, epoch, False)
