# config
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

config = {
    'DEVICE': 'cuda',
    'TRAIN_DIR': 'datasets/maps/train',
    'VAL_DIR': 'datasets/maps/val',
    'LEARNING_RATE': 2e-4,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 2,
    'IMAGE_SIZE': 256,
    'CHANNELS_IMG': 3,
    'L1_LAMBDA': 100,
    'LAMBDA_GP': 10,
    'NUM_EPOCHS': 100,
    'LOAD_MODEL': False,
    'SAVE_MODEL': False,
    'CHECKPOINT_DISC': './discriminator',
    'CHECKPOINT_GEN': './generator'}

both_transform = A.Compose([A.Resize(width=256, height=256)], additional_targets={'image0': 'image'})
transform_only_input = A.Compose([A.HorizontalFlip(0.5),
                                  A.ColorJitter(p=0.2),
                                  A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                                  ToTensorV2()])
transform_only_mask = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                                 ToTensorV2()])
