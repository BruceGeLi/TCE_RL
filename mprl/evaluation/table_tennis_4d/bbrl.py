from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Non-ctx-cov
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "table_tennis_4d_bbrl/model:version', type='model')"

    #================================================
    version_number = [542]
    epoch = 8000

    evaluation(model_str, version_number, epoch, False)
