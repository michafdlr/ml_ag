from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from utilities import MLP, MNISTDataModule, LightningModel

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=MNISTDataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=1,
        trainer_defaults={
            "max_epochs": 15,
            "callbacks": [ModelCheckpoint(monitor="val_acc", mode="max")]
        }
    )

    mlp = MLP(in_features=28*28, out_features=10, hidden_layers=1)
    lightning_model = LightningModel(mlp, learning_rate=cli.model.learning_rate)

    cli.trainer.fit(model=lightning_model, datamodule=cli.datamodule)
    cli.trainer.test(model=lightning_model, datamodule=cli.datamodule)
