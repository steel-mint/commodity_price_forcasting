from model import *
import wandb

sweep_config = {"method": "bayes"}
sweep_config["metric"] = {"name": "val_loss", "goal": "minimize"}
sweep_config["parameters"] = {
    "encoder_d": {"values": [32, 64, 128]},
    "encoder_nheads": {"values": [2, 4, 8, 16]},
    "transformer_num_layers": {"values": [2, 3, 4, 5, 6]},
    "lr": {"values": [1e-1, 1e-2, 1e-3]},
    "context_len_week": {"value": 5},
    "num_features": {"value": 18},
    "threshold": {"value": 0.5},
    "delta_p_wt": {"value": 0.5},
    "ps_wt": {"value": 0.4},
    "as_wt": {"value": 0.1},
}

sweep_id = wandb.sweep(sweep_config, project="steelmint_forcasting")


def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        datamodule = DataModule(data_config=data_config)
        wandb_logger = WandbLogger(log_model="all")
        trainer = Trainer(
            accelerator=device,
            max_epochs=1000,
            logger=wandb_logger,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=25, verbose=True)
            ],
        )
        predictor = Predictor(predictor_config=config)
        trainer.fit(model=predictor, datamodule=datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)


wandb.agent(sweep_id, sweep_train, count=5000)
