from argparse import ArgumentParser
import torch
from model import CIRModel
from validate import fashioniq_val_retrieval

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dress_type", type=str, required=True, help="dress type: [dress,shirt,toptee]"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="path of the checkpoint"
    )
    args = parser.parse_args()
    model = CIRModel()
    model.load_state_dict(torch.load(args.model_path)["CIRModel"])
    model.cuda().eval()
    print(
        fashioniq_val_retrieval(args.dress_type, model, "../data/fiq/fashionIQ_dataset")
    )
