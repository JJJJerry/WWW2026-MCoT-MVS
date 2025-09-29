import os
import json
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import CIRRDataset, collate_fn_test
from utils import device, extract_index_features
from model import CIRModel


def generate_cirr_test_submissions(file_name: str, model: CIRModel):

    model = model.float().eval()
    cirr_base_path = (
        "../data/cirr/cirr_dataset"
    )
    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset(cirr_base_path, "test1", "classic")
    index_features, index_names = extract_index_features(classic_test_dataset, model)
    relative_test_dataset = CIRRDataset(cirr_base_path, "test1", "relative")

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(
        relative_test_dataset,
        model,
        index_features,
        index_names,
    )

    submission = {"version": "rc2", "metric": "recall"}
    group_submission = {"version": "rc2", "metric": "recall_subset"}

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = Path("submission")
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(
        submissions_folder_path / f"recall_submission_{file_name}.json", "w+"
    ) as file:
        json.dump(submission, file, sort_keys=True)

    with open(
        submissions_folder_path / f"recall_subset_submission_{file_name}.json", "w+"
    ) as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(
    relative_test_dataset: CIRRDataset,
    model: CIRModel,
    index_features: torch.tensor,
    index_names: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = (
        generate_cirr_test_predictions(model, relative_test_dataset)
    )

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names
        != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(sorted_index_names), -1
        )
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
    )
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (
        (sorted_index_names[..., None] == group_members[:, None, :])
        .sum(-1)
        .astype(bool)
    )
    sorted_group_names = sorted_index_names[group_mask].reshape(
        sorted_index_names.shape[0], -1
    )

    # Generate prediction dicts
    pairid_to_predictions = {
        str(int(pair_id)): prediction[:50].tolist()
        for (pair_id, prediction) in zip(pairs_id, sorted_index_names)
    }
    pairid_to_group_predictions = {
        str(int(pair_id)): prediction[:3].tolist()
        for (pair_id, prediction) in zip(pairs_id, sorted_group_names)
    }

    return pairid_to_predictions, pairid_to_group_predictions

def generate_cirr_test_predictions(
    model: CIRModel, relative_test_dataset: CIRRDataset
) -> Tuple[torch.tensor, List[str], List[List[str]], List[str]]:

    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(
        dataset=relative_test_dataset,
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn_test,
    )

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, 1024)).to(device, non_blocking=True)
    group_members = []
    reference_names = []

    for batch in tqdm(relative_test_loader):  # Load data
        captions = batch["rel_caption"]
        batch_reference_names = batch["reference_name"]
        batch_group_members = batch["group_members"]
        batch_reference_image = batch["reference_image"]
        batch_llm_info_dict = batch['llm_info']
        batch_reference_seg_feature_list = [
            seg_feature.cuda() for seg_feature in batch["reference_seg_feature_list"]
        ]
        batch_group_members = np.array(batch_group_members).tolist()
        batch_pairs_id = batch["pair_id"]
        batch_llm_info_dict =batch['llm_info']
        
        # Compute the predicted features
        with torch.no_grad():
            batch_predicted_features = model.extract_query(
                captions,
                batch_reference_image,
                batch_reference_seg_feature_list,
                batch_llm_info_dict
            )
        
        predicted_features = torch.vstack(
            (predicted_features, F.normalize(batch_predicted_features, dim=-1))
        )
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return predicted_features, reference_names, group_members, pairs_id


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--submission_name", type=str, required=True, help="submission file name"
    )
    parser.add_argument("--model_path", required=True, type=str, help="model path")

    args = parser.parse_args()

    model = CIRModel()

    model.load_state_dict(torch.load(args.model_path)["CIRModel"])
    model.eval()
    model.cuda()
    generate_cirr_test_submissions(args.submission_name, model)


if __name__ == "__main__":
    main()
