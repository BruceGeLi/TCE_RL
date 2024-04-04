from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Entire
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "metaworld_tcp_entire/model:version', type='model')"

    #================================================
    version_number = [3406]
    epoch = 7600

    evaluation(model_str, version_number, epoch, False)
