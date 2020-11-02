from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data, width, height):
    assert data in ("train", "valid")
    assert width % 32 == 0
    assert height % 32 == 0

    if data == "train":
        return Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                transforms.ShiftScaleRotate(
                    scale_limit=0.1, shift_limit=0.0625, rotate_limit=10, p=0.2
                ),
                transforms.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.25
                ),
                transforms.Resize(width, height, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [transforms.Resize(width, height, always_apply=True),
             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ToTensorV2(),
             ]
        )
