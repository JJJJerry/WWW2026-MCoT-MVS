import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from argparse import ArgumentParser
from datetime import datetime
from statistics import mean
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CIRModel
from data_utils import FashionIQDataset, CIRRDataset, collate_fn_train
from utils import (
    save_model,
    extract_index_features,
    device
)
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics

training_hyper_params = None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_and_optimizer(model_name, clip_lr, lr, loss_weight):
    model = CIRModel(model_name=model_name, loss_weight=loss_weight)
    model = model.to(device)
    params = list(model.named_parameters())
    param_group = [
        {
            "params": [p for n, p in params if any(nd in n for nd in ["clip"])],
            "lr": clip_lr,
        },
        {
            "params": [p for n, p in params if not any(nd in n for nd in ["clip"])],
            "lr": lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_group, lr=clip_lr, weight_decay=1e-2)
    return model, optimizer


def train_fiq(
    model_name: str,
    dataset: str,
    num_epochs: int,
    clip_lr: float,
    lr: float,
    batch_size: int,
    loss_weight: float,
    validation_frequency: int,
):

    model, optimizer = create_model_and_optimizer(
        clip_lr=clip_lr, lr=lr, loss_weight=loss_weight, model_name=model_name
    )

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = f"experiments/trained_on_fiq_{training_start}"
    os.makedirs(training_path)

    with open(os.path.join(training_path, "training_hyperparameters.json"), "w+") as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    fiq_base_path = "../data/fiq/fashionIQ_dataset"
    dress_type = dataset

    relative_val_dataset = FashionIQDataset(fiq_base_path, "val", [dress_type], "relative")
    classic_val_dataset = FashionIQDataset(fiq_base_path, "val", [dress_type], "classic")
    relative_train_dataset = FashionIQDataset(fiq_base_path, "train", [dress_type], "relative")
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        collate_fn=collate_fn_train,
        drop_last=True,
        shuffle=True,
    )

    scaler = torch.cuda.amp.GradScaler()

    best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print("Training loop started")
    for epoch in range(num_epochs):
        losses = train(
            model, relative_train_loader, optimizer, scaler, epoch, num_epochs
        )
        data = {"epoch": epoch}
        for k, v in losses.items():
            data.update({k: np.array(v).mean()})

        training_log_frame = pd.concat(
            [
                training_log_frame,
                pd.DataFrame(data, index=[0]),
            ]
        )
        training_log_frame.to_csv(
            os.path.join(training_path, "train_metrics.csv"), index=False
        )

        if epoch % validation_frequency == 0:
            model.eval()
            with torch.no_grad():
                val_recall_at10, val_recall_at50, ori_recall_at10, ori_recall_at50 = (
                    compute_fiq_val_metrics(
                        relative_val_dataset, classic_val_dataset, model
                    )
                )
                results_dict = {}
                results_dict[f"{dress_type}_val_recall_at10"] = val_recall_at10
                results_dict[f"{dress_type}_val_recall_at50"] = val_recall_at50
                results_dict[f"{dress_type}_ori_recall_at10"] = ori_recall_at10
                results_dict[f"{dress_type}_ori_recall_at50"] = ori_recall_at50
                results_dict["avg"] = (val_recall_at10 + val_recall_at50) / 2

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {"epoch": epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat(
                [validation_log_frame, pd.DataFrame(data=log_dict, index=[0])]
            )
            validation_log_frame.to_csv(
                os.path.join(training_path, "validation_metrics.csv"), index=False
            )
            # Save model
            if results_dict["avg"] > best_avg_recall:
                best_avg_recall = results_dict["avg"]
                save_model(f"CIRModel", epoch, model, training_path)


def train(model: CIRModel, relative_train_loader, optimizer, scaler, epoch, num_epochs):
    losses = {}
    model.train()
    train_bar = tqdm(relative_train_loader, ascii=True)
    for idx, batch in enumerate(train_bar):  # Load a batch of triplets
        captions = batch["rel_caption"]
        reference_images = batch["reference_image"]
        target_images = batch["target_image"]
        batch_reference_seg_feature_list = [
            seg_feature.to(device) for seg_feature in batch["reference_seg_feature_list"]
        ]
        llm_info_dict = batch["llm_info"]
        optimizer.zero_grad()
        loss = 0
        with torch.cuda.amp.autocast():
            loss_dict = model.compute_loss(
                reference_images,
                captions,
                target_images,
                batch_reference_seg_feature_list,
                llm_info_dict,
            )
        for k, v in loss_dict.items():
            loss_dict[k] = v
            loss += v

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for k, v in loss_dict.items():
            if k not in losses.keys():
                losses[k] = [v.item()]
            else:
                losses[k].append(v.item())
        info = f"[{epoch}/{num_epochs}]  "
        for k, v in losses.items():
            info += f"{k}: {np.array(v).mean():.3f}  "

        train_bar.set_description(desc=info)
    return losses


def train_cirr(
    model_name: str,
    dataset: str,
    num_epochs: int,
    clip_lr: float,
    lr: float,
    batch_size: int,
    loss_weight: float,
    validation_frequency: int,
):

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = f"experiments/trained_on_cirr_{training_start}"
    os.makedirs(training_path)
    with open(
        os.path.join(training_path, "training_hyperparameters.json"), "w+"
    ) as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    cirr_base_path = "../data/cirr/cirr_dataset"
    relative_val_dataset = CIRRDataset(cirr_base_path, "val", "relative")
    classic_val_dataset = CIRRDataset(cirr_base_path, "val", "classic")

    relative_train_dataset = CIRRDataset(cirr_base_path, "train", "relative")

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_train,
        drop_last=True,
        shuffle=True,
    )

    model, optimizer = create_model_and_optimizer(
        model_name=model_name, clip_lr=clip_lr, lr=lr, loss_weight=loss_weight
    )
    scaler = torch.cuda.amp.GradScaler()

    best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print("Training loop started")
    for epoch in range(num_epochs):
        model.train()
        losses = train(
            model, relative_train_loader, optimizer, scaler, epoch, num_epochs
        )
        for g in optimizer.param_groups:
            g["lr"] *= 0.95
        # Training CSV logging
        data = {"epoch": epoch}
        for k, v in losses.items():
            data.update({k: np.array(v).mean()})

        training_log_frame = pd.concat(
            [
                training_log_frame,
                pd.DataFrame(data, index=[0]),
            ]
        )
        training_log_frame.to_csv(
            os.path.join(training_path, "train_metrics.csv"), index=False
        )
        if epoch % validation_frequency == 0:
            model.eval()
            model.float()
            with torch.no_grad():
                val_index_features, val_index_names = extract_index_features(
                    classic_val_dataset, model
                )
                results = compute_cirr_val_metrics(
                    relative_val_dataset, model, val_index_features, val_index_names
                )
            (
                group_recall_at1,
                group_recall_at2,
                group_recall_at3,
                recall_at1,
                recall_at5,
                recall_at10,
                recall_at50,
            ) = results

            results_dict = {
                "group_recall_at1": group_recall_at1,
                "group_recall_at2": group_recall_at2,
                "group_recall_at3": group_recall_at3,
                "recall_at1": recall_at1,
                "recall_at5": recall_at5,
                "recall_at10": recall_at10,
                "recall_at50": recall_at50,
                "mean(R@5+R_s@1)": (group_recall_at1 + recall_at5) / 2,
                "arithmetic_mean": mean(results),
            }

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {"epoch": epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat(
                [validation_log_frame, pd.DataFrame(data=log_dict, index=[0])]
            )
            validation_log_frame.to_csv(
                os.path.join(training_path, "validation_metrics.csv"), index=False
            )

            # Save model
            if results_dict["arithmetic_mean"] > best_arithmetic:
                best_arithmetic = results_dict["arithmetic_mean"]
                save_model(f"CIRModel", epoch, model, training_path)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cirr",
        help="['cirr','dress','shirt','toptee']",
    )
    parser.add_argument("--loss_weight", default=10.0, type=float, help="logits weight")
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="number training epochs"
    )
    parser.add_argument(
        "--clip_lr", default=1e-6, type=float, help="CLIP learning rate"
    )
    parser.add_argument("--lr", default=1e-6, type=float, help="other learning rate")
    parser.add_argument("--model_name", default="ViT-H-14", type=str)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--validation-frequency",
        default=1,
        type=int,
        help="Validation frequency expressed in epochs",
    )
    parser.add_argument(
        "--seed",
        default=124,
        type=int,
        help="random seed",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    global training_hyper_params

    training_hyper_params = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "num_epochs": args.num_epochs,
        "clip_lr": args.clip_lr,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "loss_weight": args.loss_weight,
        "validation_frequency": args.validation_frequency,
    }

    if args.dataset.lower() == "cirr":
        train_cirr(**training_hyper_params)
    elif args.dataset.lower() in ["dress", "shirt", "toptee"]:
        train_fiq(**training_hyper_params)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
