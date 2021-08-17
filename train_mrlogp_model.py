"""
This program provides users with the procedures for easily doing the hyperparameter scan, cross validation, and transfer training using MRLogP package. 
This also allows users to use their own data set and hyperparameters in the procedures above. Other than training procedures, molecular logPs can be 
directly predicted by MRLogP using this program.

By Yan-Kai Chen- justin9300454@gmail.com
"""
from mrlogp import MRlogP
import argparse
from pathlib import Path
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRlogP logP predictor")
    subparsers = parser.add_subparsers(dest='subcmd', help='subcommands', metavar='SUBCOMMAND')
    subparsers.required = True

    #hyperparameter scan fuction
    f1_parser = subparsers.add_parser("para_scan", help="Perform hyperparameter scan with given a set of hyperparameter combinations using gird search")
    f1_parser.add_argument("-ld", "--large_dataset", help="Dataset containing MRlogP descriptors and logPs used for model training. Default to the 500k_DL set in the manuscript", default=Path("./datasets")/Path("ds_descriptors_500K_DL.csv"), nargs='?', dest="ld")
    f1_parser.add_argument("-para_dict", "--hyperparameter_dict", help="File contains a dictionary includes set hyperparameters used to perform hyperparameter scan", default=Path("./example")/Path("hyperparameter_list.txt"), nargs='?', dest="para_dict")
    f1_parser.add_argument("-val_split", "--training_test_split", help="Fraction for splitting a subset from training set for validation", type=float, default=0.1, nargs='?', dest="val_split")

    #cross validation
    f2_parser = subparsers.add_parser("cv", help="Perform cross validation with given a set of hyperparameter combinations")
    f2_parser.add_argument("-ld", "--large_dataset", help="Dataset containing MRlogP descriptors and logPs used for model training. Default to the 500k_DL set in the manuscript", default=Path("./datasets")/Path("ds_descriptors_500K_DL.csv"), nargs='?', dest="ld")
    f2_parser.add_argument("-para_dict", "--hyperparameter_dict", help="File contains a dictionary includes set hyperparameters used to perform cross validation", default=Path("./example")/Path("hyperparameter_list.txt"), nargs='?', dest="para_dict")
    f2_parser.add_argument("-cv", "--n_fold_cross_validation", help="Number of folds for cross validation", type=int, default=10, nargs='?', dest="cv")

    #final training 
    f3_parser = subparsers.add_parser("f_train", help="Train the neural network model with the given hyperparameters against the full training set and then evaluate against test sets")
    f3_parser.add_argument("-ld", "--large_dataset", help="Dataset containing MRlogP descriptors and logPs used for model training. Default to the 500k_DL set in the manuscript", default=Path("./datasets")/Path("ds_descriptors_500K_DL.csv"), nargs='?', dest="ld")
    f3_parser.add_argument("-md", "--small_precise_dataset", help="Small set of MRlogP descriptors containing highly accurate (measured) used to evaluate the model at the end of the training", default=Path("./datasets")/Path("ds_descriptors_martel_DL.csv"), nargs='?', dest="md")
    f3_parser.add_argument("-rd", "--reaxys_dataset", help="2nd test set used to evaluate the model at the end of the training", default=Path("./datasets")/Path("ds_descriptors_reaxys_DL.csv"), nargs='?', dest="rd")
    f3_parser.add_argument("-pd", "--physprop_dataset", help="3rd test set used to evaluate the model at the end of the training", default=Path("./datasets")/Path("ds_descriptors_physprop_DL.csv"), nargs='?', dest="pd")
    f3_parser.add_argument("-mod_file", "--model_file", help="Model file of an exsisting model to be evaluated against the test sets", default=Path("./example/models")/Path("mrlogp_model.hdf5"), nargs='?', dest="mod_file")

    #transfer learning
    f4_parser = subparsers.add_parser("t_train", help="Perform transfer learning on a pre-trained model with given a set of transfer learning hyperparameters")
    f4_parser.add_argument("-ld", "--large_dataset", help="Dataset containing MRlogP descriptors and logPs used for model training. Default to the 500k_DL set in the manuscript", default=Path("./datasets")/Path("ds_descriptors_500K_DL.csv"), nargs='?', dest="ld")
    f4_parser.add_argument("-md", "--small_precise_dataset", help="Dataset containing high quality measured logPs used to futher tweak the pre-trained model for higher prediction performance", default=Path("./datasets")/Path("ds_descriptors_martel_DL.csv"), nargs='?', dest="md")
    f4_parser.add_argument("-rd", "--reaxys_dataset", help="Test set used to evaluate the model at the end of transfer learning", default=Path("./datasets")/Path("ds_descriptors_reaxys_DL.csv"), nargs='?', dest="rd")
    f4_parser.add_argument("-pd", "--physprop_dataset", help="2nd test set used to evaluate the model at the end of transfer learning", default=Path("./datasets")/Path("ds_descriptors_physprop_DL.csv"), nargs='?', dest="pd")
    f4_parser.add_argument("-mod_file", "--model_file", help="The path of the pre-trained model needed for transfer learing", default=Path("./example/models")/Path("mrlogP_model_consensus.hdf5"), nargs='?', dest="mod_file")

    #predict logP
    f5_parser = subparsers.add_parser("pred_logp", help="Predict logP on query compounds using given model")
    f5_parser.add_argument("-ld", "--large_dataset", help="Dataset containing MRlogP descriptors and logPs used for model training. Default to the 500k_DL set in the manuscript", default=Path("./datasets")/Path("ds_descriptors_500K_DL.csv"), nargs='?', dest="ld")
    f5_parser.add_argument("-q_mol", "--query_file", help="Descriptor file of query compounds for logP prediction", default=Path("./example/compounds")/Path("sample_5cpd.csv"), nargs='?', dest="q_mol")
    f5_parser.add_argument("-mod_file", "--model_file", help="The path of the model used as the predictor performing logP prediction", default=Path("./example/models")/Path("mrlogp_model.hdf5"), nargs='?', dest="mod_file")

    #parser.add_argument("working_dir", help="The directory where the relevant output files will save", default={a})
    args = parser.parse_args()
    mrlogp=MRlogP()

    #Hyperparameter scan:
    if args.subcmd == "para_scan":
        if args.para_dict != None:
            try: args.para_dict = eval(open(args.para_dict, 'r').read())
            except: 
                print (f"{args.para_dict} is not a valid file, please see a hyperparameter example in the example folder")
                sys.exit(-1)
        print (args.ld, args.para_dict, args.val_split)
        mrlogp.hyerparameter_scan(args.ld, args.para_dict, args.val_split)

    #10-fold cross validation:
    if args.subcmd == "cv":
        mrlogp.cv(args.ld, args.para_dict, args.cv)

    #Final training against full large dataset
    if args.subcmd == "f_train":
        mrlogp.final_train(args.ld, args.md, args.rd, args.pd, args.mod_file)
    
    #Transfer learning 
    if args.subcmd == "t_train":
        mrlogp.transfer_learning(args.ld, args.md, args.rd, args.pd, args.mod_file)

    #Predict molecular logP
    if args.subcmd == "pred_logp":
        mrlogp.predict_logp(args.ld, args.q_mol, args.mod_file)