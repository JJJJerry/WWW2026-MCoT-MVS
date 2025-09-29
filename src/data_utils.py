import json
from typing import List
import torch
import os
import PIL
import PIL.Image
import string
import cv2
import numpy as np
from torch.utils.data import Dataset


def load_mask(path):
    mask_array = np.load(path)["array"]
    if len(mask_array.shape) == 4:
        mask_array = mask_array.reshape(
            mask_array.shape[0], mask_array.shape[2], mask_array.shape[3]
        )
    return mask_array


def draw_text(img, point, text, drawType="custom"):
    """
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    """
    fontScale = 0.7
    thickness = 5
    text_thickness = 2
    bg_color = (255, 255, 255)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(
            img,
            (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
            (text_loc[0] + text_size[0], text_loc[1] + text_size[1]),
            bg_color,
            -1,
        )
        # draw score value
        cv2.putText(
            img,
            str(text),
            (text_loc[0], text_loc[1] + baseline),
            fontFace,
            fontScale,
            (255, 0, 0),
            text_thickness,
            8,
        )
    elif drawType == "simple":
        cv2.putText(img, "%d" % (text), point, fontFace, 0.5, (255, 0, 0))
    return img


def draw_text_line(img, point, text_line: str, drawType="custom"):
    """
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    """
    fontScale = 0.7
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split(", ")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(
        str(text_line), fontFace, fontScale, thickness
    )
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img


def collate_fn_train(batch):
    out = {}
    out["reference_image"] = [item["reference_image"] for item in batch]
    out["reference_seg_feature_list"] = [
        item["reference_seg_feature"] for item in batch
    ]
    out["target_image"] = [item["target_image"] for item in batch]
    out["rel_caption"] = [item["rel_caption"] for item in batch]
    out["llm_info"] = {
        "target": [item["llm_info"]["target"] for item in batch],
        "retained": [item["llm_info"]["retained"] for item in batch],
        "deleted": [item["llm_info"]["deleted"] for item in batch],
    }
    return out


def collate_fn_val(batch):
    out = {}
    out["reference_image"] = [item["reference_image"] for item in batch]
    out["reference_name"] = [item["reference_name"] for item in batch]
    out["reference_seg_feature_list"] = [
        item["reference_seg_feature"] for item in batch
    ]
    out["target_hard_name"] = [item["target_hard_name"] for item in batch]
    out["rel_caption"] = [item["rel_caption"] for item in batch]
    out["group_members"] = [item["group_members"] for item in batch]
    out["llm_info"] = {
        "target": [item["llm_info"]["target"] for item in batch],
        "retained": [item["llm_info"]["retained"] for item in batch],
        "deleted": [item["llm_info"]["deleted"] for item in batch],
    }
    return out


def collate_fn_val_fiq(batch):
    out = {}
    out["reference_image"] = [item["reference_image"] for item in batch]
    out["reference_name"] = [item["reference_name"] for item in batch]
    out["reference_seg_feature_list"] = [
        item["reference_seg_feature"] for item in batch
    ]
    out["target_hard_name"] = [item["target_name"] for item in batch]
    out["rel_caption"] = [item["rel_caption"] for item in batch]
    out["llm_info"] = {
        "target": [item["llm_info"]["target"] for item in batch],
        "retained": [item["llm_info"]["retained"] for item in batch],
        "deleted": [item["llm_info"]["deleted"] for item in batch],
    }
    return out

def collate_fn_classic_fiq(batch):
    out = {}
    out["image"] = [item["image"] for item in batch]
    out["image_name"] = [item["image_name"] for item in batch]
    return out

def collate_fn_test(batch):
    out = {}
    out["reference_image"] = [item["reference_image"] for item in batch]
    out["reference_seg_feature_list"] = [
        item["reference_seg_feature"] for item in batch
    ]
    out["reference_name"] = [item["reference_name"] for item in batch]
    out["rel_caption"] = [item["rel_caption"] for item in batch]
    out["group_members"] = [item["group_members"] for item in batch]
    out["pair_id"] = [item["pair_id"] for item in batch]
    out["llm_info"] = {
        "target": [item["llm_info"]["target"] for item in batch],
        "retained": [item["llm_info"]["retained"] for item in batch],
        "deleted": [item["llm_info"]["deleted"] for item in batch],
    }
    return out


class FashionIQDataset(Dataset):
    def __init__(
        self, fiq_base_path: str, split: str, dress_types: List[str], mode: str
    ):
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.seg_feature_dir='../data/fiq/segment/seg_features_vit-h_patch'
        
        self.base_path = fiq_base_path
        self.image_base_path = os.path.join(fiq_base_path,'resized_image')
        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ["test", "train", "val"]:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ["dress", "shirt", "toptee"]:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.triplets_dir = "../data/fiq/llm_data"
        
        self.triplets: List[dict] = []
        self.target_keywords = {}
        for dress_type in dress_types:
            with open(os.path.join(self.triplets_dir,f"cap_llm.{dress_type}.{split}.json")) as f:
                self.triplets.extend(json.load(f))
            with open(os.path.join(fiq_base_path,f"keywords_in_mods_{dress_type}.json"),"r") as f:
                self.target_keywords.update(json.load(f))
        self.correction_dict = {}
        for dress_type in dress_types:
            with open(os.path.join(self.base_path,f"correction_dict_{dress_type}.json")) as f:
                self.correction_dict.update(json.load(f))

        self.image_names: list = []
        for dress_type in dress_types:
            with open(os.path.join(self.base_path,"image_splits",f"split.{dress_type}.{split}.json")) as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")
        
    def get_image_by_name(self,name):
        image_path = os.path.join(self.image_base_path,f"{name}.png")
        image = PIL.Image.open(image_path).convert("RGB")
        return image
    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                mod_str = self.concat_text(self.triplets[index]["captions"])
                reference_name = self.triplets[index]["candidate"]
                reference_image_path = os.path.join(self.image_base_path,f"{reference_name}.png")
                target_name = self.triplets[index]["target"]
                key_words = self.target_keywords[reference_name + "_" + target_name][-1]
                llm_info = {
                    "target":self.triplets[index]["ModifiedImageCaption"],
                    "retained": self.triplets[index]["Retained"],
                    "deleted": self.triplets[index]["Removed"],
                }
                reference_image = self.get_written_image(
                    reference_image_path, key_words
                )
                if self.split == "train":
                    out = {}
                    out["reference_image"] = reference_image
                    out["llm_info"] = llm_info
                    out["target_image"] = self.get_image_by_name(target_name)
                    out["rel_caption"] = mod_str
                    out['reference_seg_feature'] = torch.load(os.path.join(self.seg_feature_dir,reference_name,"seg_feature.pt"), map_location='cpu')
                    return out
                elif self.split == "val":
                    out = {}
                    out["llm_info"] = llm_info
                    out["reference_image"] = reference_image
                    out["reference_name"] = reference_name
                    out["target_name"] = target_name
                    out["rel_caption"] = mod_str
                    out['reference_seg_feature'] = torch.load(os.path.join(self.seg_feature_dir,reference_name,"seg_feature.pt"), map_location='cpu')
                    return out
                
            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image = self.get_image_by_name(image_name)
                out={}
                out['image_name'] = image_name
                out['image'] = image
                return out
            
            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def correct_text(self, text):
        trans = str.maketrans({key: " " for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join(
            [
                self.correction_dict.get(word) if word in self.correction_dict else word
                for word in tokens
            ]
        )

        return text

    def get_written_image(self, image_path, key_words):
        candidate_img = cv2.imread(image_path)
        candidate_img = cv2.resize(candidate_img, (512, 512))
        written_img = draw_text_line(candidate_img, (15, 15), key_words)
        written_img = PIL.Image.fromarray(
            cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB)
        ).convert("RGB")
        return written_img

    def concat_text(self, captions):
        text = "{} and {}".format(
            self.correct_text(captions[0]),
            self.correct_text(captions[1]),
        )
        return text

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

class CIRRDataset(Dataset):
    def __init__(self, cirr_base_path, split: str, mode: str):
        self.mode = mode
        self.split = split
        self.seg_feature_dir='../data/cirr/segment/seg_features_vit-h_patch'
        self.cirr_base_path = cirr_base_path
        if split not in ["test1", "train", "val"]:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(
            os.path.join(cirr_base_path, "cirr", "captions", f"cap.rc2.{split}.json"),
            "r",
        ) as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(
            os.path.join(
                cirr_base_path, "cirr", "image_splits", f"split.rc2.{split}.json"
            ),
            "r",
        ) as f:
            self.name_to_relpath = json.load(f)

        with open(
            f"../data/cirr/llm_data/llm_info_cirr_{split}.json",
            "r",
        ) as f:
            self.llm_caption = json.load(f)

        with open(os.path.join(cirr_base_path,f"keywords_in_mods_cirr_{split}.json"),"r") as f:
            self.target_keywords = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def get_written_image(self, image_path, key_words):
        candidate_img = cv2.imread(image_path)
        written_img = draw_text_line(candidate_img, (15, 15), key_words)
        written_img = PIL.Image.fromarray(
            cv2.cvtColor(written_img, cv2.COLOR_BGR2RGB)
        ).convert("RGB")
        return written_img

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                group_members = self.triplets[index]["img_set"]["members"]
                reference_name = self.triplets[index]["reference"]
                rel_caption = self.triplets[index]["caption"]
                reference_image_path = os.path.join(
                    self.cirr_base_path, self.name_to_relpath[reference_name]
                )
                reference_seg_feature = torch.load(os.path.join(self.seg_feature_dir,reference_name,"seg_feature.pt"), map_location='cpu')
                if self.split == "train":
                    out = {}
                    target_hard_name = self.triplets[index]["target_hard"]
                    key_words = self.target_keywords[
                    reference_name + "+" + target_hard_name
                    ][-1]
                    llm_info = {
                        "target": self.llm_caption[reference_name + target_hard_name][
                            "ModifiedImageCaption"
                        ],
                        "retained": self.llm_caption[reference_name + target_hard_name][
                            "Retained"
                        ],
                        "deleted": self.llm_caption[reference_name + target_hard_name][
                            "Removed"
                        ],
                    }
                    target_image_path = os.path.join(
                        self.cirr_base_path, self.name_to_relpath[target_hard_name]
                    )
                    reference_image = self.get_written_image(
                        reference_image_path, key_words
                    )
                    out["reference_image"] = reference_image
                    out["target_image"] = PIL.Image.open(target_image_path)
                    out["rel_caption"] = rel_caption
                    out['reference_seg_feature'] = reference_seg_feature
                    out["llm_info"] = llm_info
                    return out

                elif self.split == "val":
                    out = {}
                    target_hard_name = self.triplets[index]["target_hard"]
                    key_words = self.target_keywords[
                    reference_name + "+" + target_hard_name
                    ][-1]
                    llm_info = {
                        "target": self.llm_caption[reference_name + target_hard_name][
                            "ModifiedImageCaption"
                        ],
                        "retained": self.llm_caption[reference_name + target_hard_name][
                            "Retained"
                        ],
                        "deleted": self.llm_caption[reference_name + target_hard_name][
                            "Removed"
                        ],
                    }
                    reference_image = self.get_written_image(
                        reference_image_path, key_words
                    )
                    out["reference_image"] = reference_image
                    out["reference_name"] = reference_name
                    out["target_hard_name"] = target_hard_name
                    out["rel_caption"] = rel_caption
                    out["group_members"] = group_members
                    out["llm_info"] = llm_info
                    out['reference_seg_feature'] = reference_seg_feature
                    return out

                elif self.split == "test1":
                    pair_id = self.triplets[index]["pairid"]
                    out = {}
                    out["pair_id"] = pair_id
                    key = str(pair_id) + reference_name
                    key_words = self.target_keywords[
                        str(pair_id) + "+" + reference_name
                    ][-1]
                    llm_info = {
                        "target": self.llm_caption[key]["ModifiedImageCaption"],
                        "retained": self.llm_caption[key]["Retained"],
                        "deleted": self.llm_caption[key]["Removed"],
                    }
                    reference_image = self.get_written_image(
                        reference_image_path, key_words
                    )
                    out['reference_seg_feature'] = reference_seg_feature
                    out["reference_name"] = reference_name
                    out["reference_image"] = reference_image
                    out["rel_caption"] = rel_caption
                    out["group_members"] = group_members
                    out["llm_info"] = llm_info
                    return out

            elif self.mode == "classic":
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = os.path.join(
                    self.cirr_base_path, self.name_to_relpath[image_name]
                )
                im = PIL.Image.open(image_path).convert("RGB")
                image = im
                out = {}
                out["image_name"] = image_name
                out["image"] = image
                return out

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
