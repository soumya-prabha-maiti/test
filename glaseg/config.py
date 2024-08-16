class SegConfig:
    TRAIN_BATCH_SIZE: int = 2
    VAL_BATCH_SIZE: int = 2
    TEST_BATCH_SIZE: int = 1
    ROI: tuple = (512, 512)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    DEVICE = None
    