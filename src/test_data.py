import hydra
from omegaconf import DictConfig
import torch
from data import ContrastiveSegDataset
from utils import get_transform
import torchvision.transforms as T

@hydra.main(config_path="configs", config_name="train_config.yaml")
def test(cfg: DictConfig):
    pytorch_data_dir = "/Users/navneetbavineni/STEGO/src/pytorch_data_dir"
    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(cfg.crop_ratio, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])
    
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )
    
    print("Dataset created, len:", len(train_dataset))
    item = train_dataset[0]
    print("Got item, keys:", item.keys())

if __name__ == "__main__":
    test()
