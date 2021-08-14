# MRlogP
Neural network-based logP prediction for druglike molecules 

MRlogP is a neural network-based predictor of logP using transfer learning techniques, first learning on a large amount of low accuracy predicted logP values before finally tuning our model using a small, accurate dataset of 244 druglike compounds. MRlogP is capable of outperforming state of the art freely available logP prediction methods for druglike small molecule, achieves an average RMSE of 0.988 and 0.715 against druglike molecules from Reaxys and PHYSPROP. 

All associated code for MRlogP is freely available on GitHub to run locally and a simple and easy to use online web-based version is also provided at https://similaritylab.bio.ed.ac.uk/mrlogp. 

[MRlogP structure](https://raw.githubusercontent.com/JustinYKC/MRlogP/justin_upload/20210723_dnn_structure.png)

# Requirements
MRlogP requires the following packages:
- rdkit (2019.09.x.x)
- OpenBabel (2.4.x) 
- numpy (1.19.x)
- pandas (0.24.x)
- scikit-learn (0.20.x)
- TensorFlow (2.2.x)
- Keras (2.2.x)

# Authors
MRlogP was written by Yan-Kai Chen (justin9300454@gmail.com)