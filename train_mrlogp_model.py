from mrlogp import MRlogP
import argparse
from pathlib import Path

import mrlogp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRlogP logP predictor")
    parser.add_argument("large_dataset", help="Large set of MRlogP descriptors containing predicted logPs, names the 500k_DL set in the manuscript", default=Path("data")/Path("ds_descriptors_500K_DL.csv"))
    parser.add_argument("small_precise_dataset", help="Small set of MRlogP descriptors containing highly accurate (measured) logP values", default=Path("data")/Path("ds_descriptors_martel_DL.csv"))
    parser.add_argument("reaxys_dataset", help="Physprop_DL dataset", default=Path("data")/Path("ds_descriptors_reaxys_DL.csv"))
    parser.add_argument("physprop_dataset", help="Physprop_DL dataset",default=Path("data")/Path("ds_descriptors_physprop_DL.csv"))
    parser.add_argument("training_test_split", type=float, default=0.1)

    args = parser.parse_args()
    mrlogp=MRlogP()
    mrlogp.train(args.large_dataset, args.small_precise_dataset, args.reaxys_dataset,args.physprop_dataset, args.training_test_split)


