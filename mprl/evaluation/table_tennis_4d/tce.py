from mprl.mp_exp import evaluation

if __name__ == "__main__":
    # Entire
    model_str = "artifact = run.use_artifact('WANDB_USERNAME/" \
                "table_tennis_4d_tcp/model:version', type='model')"

    #================================================
    version_number = [1686]
    epoch = 12000

    evaluation(model_str, version_number, epoch, False)
