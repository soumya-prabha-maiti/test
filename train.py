from glaseg.model.net import SegmentationModel
from glaseg.module.lightning_module import SegmentationDataModule
import lightning as pl


if __name__ == "__main__":
    # Train the model
    data_module = SegmentationDataModule(data_dir="data")
    model = SegmentationModel(num_classes=2)  # 2 classes in this example
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
