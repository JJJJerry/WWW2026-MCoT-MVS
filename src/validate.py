import multiprocessing
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    FashionIQDataset,
    CIRRDataset,
    collate_fn_val,
    collate_fn_val_fiq,
    collate_fn_classic_fiq,
)

from utils import device
from model import CIRModel


def compute_fiq_val_metrics(
    relative_val_dataset: FashionIQDataset,
    classic_val_dataset: FashionIQDataset,
    model: CIRModel,
) -> Tuple[float, float]:
    # Generate predictions
    (
        predicted_features,
        target_names,
        index_features,
        index_names,
        index_features_ori,
        index_names_ori,
    ) = generate_fiq_val_predictions(relative_val_dataset, classic_val_dataset, model)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    def get_metrics(query_feature, target_name, index_feature, index_name):
        # Normalize the index features
        index_feature = F.normalize(index_feature, dim=-1).float()

        # Compute the distances and sort the results
        distances = 1 - query_feature @ index_feature.T
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
        sorted_index_names = np.array(index_name)[sorted_indices]

        # Compute the ground-truth labels wrt the predictions
        labels = torch.tensor(
            sorted_index_names
            == np.repeat(np.array(target_name), len(index_name)).reshape(
                len(target_name), -1
            )
        )
        assert torch.equal(
            torch.sum(labels, dim=-1).int(), torch.ones(len(target_name)).int()
        )

        # Compute the metrics
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

        return recall_at10, recall_at50

    val_split_recall_at10, val_split_recall_at50 = get_metrics(
        predicted_features, target_names, index_features, index_names
    )
    ori_split_recall_at10, ori_split_recall_at50 = get_metrics(
        predicted_features, target_names, index_features_ori, index_names_ori
    )
    return (
        val_split_recall_at10,
        val_split_recall_at50,
        ori_split_recall_at10,
        ori_split_recall_at50,
    )


def generate_fiq_val_predictions(
    relative_val_dataset: FashionIQDataset,
    classic_val_dataset: FashionIQDataset,
    model: CIRModel,
) -> Tuple[torch.tensor, List[str]]:
    print(
        f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions"
    )

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn_val_fiq,
        shuffle=False,
    )
    classic_val_loader = DataLoader(
        dataset=classic_val_dataset,
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn_classic_fiq,
        shuffle=False,
    )
    index_features = torch.empty((0, 1024)).to(device)
    index_features_ori = torch.empty((0, 1024)).to(device)
    predicted_features = torch.empty((0, 1024)).to(device)
    index_names = []
    index_names_ori = []
    target_names = []
    reference_names = []
    for batch in tqdm(relative_val_loader):  # Load data
        captions = batch["rel_caption"]
        batch_reference_image = batch["reference_image"]
        batch_target_name = batch["target_hard_name"]
        batch_llm_info_dict = batch["llm_info"]
        batch_reference_name = batch["reference_name"]
        batch_reference_seg_feature_list = [
            seg_feature.cuda() for seg_feature in batch["reference_seg_feature_list"]
        ]
        with torch.no_grad():
            batch_predicted_features = model.extract_query(
                captions,
                batch_reference_image,
                batch_reference_seg_feature_list,
                batch_llm_info_dict,
            )
        predicted_features = torch.vstack(
            (predicted_features, batch_predicted_features)
        )
        target_names.extend(batch_target_name)
        reference_names.extend(batch_reference_name)

    for i in range(len(target_names)):
        reference_name = reference_names[i]
        target_name = target_names[i]
        if reference_name not in index_names:
            index_names.append(reference_name)
        if target_name not in index_names:
            index_names.append(target_name)

    for name in tqdm(index_names):
        image = relative_val_dataset.get_image_by_name(name)
        with torch.no_grad():
            index_feature = model.extract_target(image)  # (1,D)
        index_features = torch.vstack((index_features, index_feature))

    for batch in tqdm(classic_val_loader):
        images = batch["image"]
        image_names = batch["image_name"]
        with torch.no_grad():
            index_feature = model.extract_target(images)  # (1,D)
        index_features_ori = torch.vstack((index_features_ori, index_feature))
        index_names_ori.extend(image_names)
    return (
        predicted_features,
        target_names,
        index_features,
        index_names,
        index_features_ori,
        index_names_ori,
    )


def fashioniq_val_retrieval(dress_type: str, model: CIRModel, fiq_base_path: str):
    model = model.float().eval()
    relative_val_dataset = FashionIQDataset(
        fiq_base_path, "val", [dress_type], "relative"
    )
    classic_val_dataset = FashionIQDataset(
        fiq_base_path, "val", [dress_type], "classic"
    )
    return compute_fiq_val_metrics(relative_val_dataset, classic_val_dataset, model)


def compute_cirr_val_metrics(
    relative_val_dataset: CIRRDataset,
    model: CIRModel,
    index_features: torch.tensor,
    index_names: List[str],
) -> Tuple[float, float, float, float, float, float, float]:
    # Generate predictions
    query_features, reference_names, target_names, group_members = (
        generate_cirr_val_predictions(model, relative_val_dataset)
    )

    print("Compute CIRR validation metrics")

    query_features = query_features.cpu().numpy()
    index_features = index_features.cpu().numpy()
    distances = 1 - query_features.dot(index_features.T)
    sorted_indices = np.argsort(distances, axis=1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names
        != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )

    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
    )

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names) - 1).reshape(
            len(target_names), -1
        )
    )

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (
        (sorted_index_names[..., None] == group_members[:, None, :])
        .sum(-1)
        .astype(bool)
    )
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int()
    )
    assert torch.equal(
        torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int()
    )

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return (
        group_recall_at1,
        group_recall_at2,
        group_recall_at3,
        recall_at1,
        recall_at5,
        recall_at10,
        recall_at50,
    )


def generate_cirr_val_predictions(
    model: CIRModel, relative_val_dataset: CIRRDataset
) -> Tuple[torch.tensor, List[str], List[str], List[List[str]]]:

    print("Compute CIRR validation predictions")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_val,
    )

    # Get a mapping from index names to index features

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, 1024)).to(device)
    target_names = []
    group_members = []
    reference_names = []

    for batch in tqdm(relative_val_loader):  # Load data
        batch_group_members = batch["group_members"]
        batch_group_members = np.array(batch_group_members).tolist()
        captions = batch["rel_caption"]
        batch_reference_image = batch["reference_image"]
        batch_reference_names = batch["reference_name"]
        batch_target_names = batch["target_hard_name"]
        batch_llm_info_dict = batch["llm_info"]
        batch_reference_seg_feature_list = [
            seg_feature.cuda() for seg_feature in batch["reference_seg_feature_list"]
        ]
        with torch.no_grad():
            batch_predicted_features = model.extract_query(
                captions,
                batch_reference_image,
                batch_reference_seg_feature_list,
                batch_llm_info_dict,
            )
        predicted_features = torch.vstack(
            (predicted_features, batch_predicted_features)
        )
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members
