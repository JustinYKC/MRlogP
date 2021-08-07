from mrlogp import MRlogP
import argparse
from pathlib import Path
import mrlogp as mr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRlogP logP predictor")
    parser.add_argument("large_dataset", help="Large set of MRlogP descriptors containing predicted logPs, names the 500k_DL set in the manuscript", default=Path("../encapsulated_version/data")/Path("ds_descriptors_500K_DL.csv"), nargs='?')
    parser.add_argument("small_precise_dataset", help="Small set of MRlogP descriptors containing highly accurate (measured) logP values", default=Path("../encapsulated_version/data")/Path("ds_descriptors_martel_DL.csv"), nargs='?')
    parser.add_argument("reaxys_dataset", help="Physprop_DL dataset", default=Path("../encapsulated_version/data")/Path("ds_descriptors_reaxys_DL.csv"), nargs='?')
    parser.add_argument("physprop_dataset", help="Physprop_DL dataset", default=Path("../encapsulated_version/data")/Path("ds_descriptors_physprop_DL.csv"), nargs='?')
    parser.add_argument("training_test_split", help="Fraction for splitting a subset from training set for validation", type=float, default=0.1, nargs='?')
    parser.add_argument("cv", help="Number of folds for cross validation", type=int, default=10, nargs='?')
    parser.add_argument("model_file", help="Model file for an exsisting model", default=Path("../../../final_training_against_whole_trainingset/3")/Path("model-79-6-302_endTraining_beforeTL.hdf5"), nargs='?')
    #parser.add_argument("working_dir", help="The directory where the relevant output files will save", default={a})
    args = parser.parse_args()
    mrlogp=MRlogP()
    #mrlogp.train(args.large_dataset, args.small_precise_dataset, args.reaxys_dataset, args.physprop_dataset, args.training_test_split)
    
    #Hyperparameter scan:
    #mrlogp.hyerparameter_scan(args.large_dataset)

    #10-fold cross validation
    #mrlogp.cv(args.large_dataset, args.cv)

    #Final training against full large dataset
    #mrlogp.final_train(args.large_dataset, args.small_precise_dataset, args.reaxys_dataset, args.physprop_dataset, args.model_file)
    
    #Transfer learning 
    mrlogp.transfer_learning(args.large_dataset, args.small_precise_dataset, args.reaxys_dataset, args.physprop_dataset, args.model_file)