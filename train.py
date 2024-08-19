import lightning as L

from glaseg.model.net import SegmentationModel
from glaseg.module.lightning_module import SegmentationDataModule

if __name__ == "__main__":
    # Train the model
    data_module = SegmentationDataModule(data_dir="data")
    model = SegmentationModel(num_classes=2)  # 2 classes in this example
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)
