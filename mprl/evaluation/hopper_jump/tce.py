from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Entire
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "hopper_jump_tcp/model:version', type='model')"

    #================================================
    version_number = [646]
    epoch = 3000

    evaluation(model_str, version_number, epoch, False)
