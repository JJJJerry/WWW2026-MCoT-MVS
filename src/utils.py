import multiprocessing
import os
from typing import Union, Tuple, List

import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_index_features(
    dataset: Union[CIRRDataset, FashionIQDataset], model
) -> Tuple[torch.tensor, List[str]]:

    def collate_fn_local(batch):
        out = {}
        out["image"] = [item["image"] for item in batch]
        out["name"] = [item["image_name"] for item in batch]
        return out

    feature_dim = 1024
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn_local,
    )
    index_features = torch.empty((0, feature_dim)).to(device)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(
            f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features"
        )
    model.eval()
    for batch in tqdm(classic_val_loader):
        images = batch["image"]
        names = batch["name"]
        with torch.no_grad():
            batch_features = model.extract_target(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)

    return index_features, index_names


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: str):
    models_path = os.path.join(training_path, "saved_models")
    os.makedirs(models_path, exist_ok=True)
    model_name = model_to_save.__class__.__name__
    torch.save(
        {
            "epoch": cur_epoch,
            model_name: model_to_save.state_dict(),
        },
        os.path.join(models_path, f"{name}.pt"),
    )
