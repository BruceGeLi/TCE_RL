from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Entire
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "box_random_tcp_entire/model:version', type='model')"

    #================================================
    version_number = [839]
    epoch = 7500

    evaluation(model_str, version_number, epoch, False)
