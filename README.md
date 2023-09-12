# Pisces: A multi-modal data augmentation approach for drug combination synergy prediction
This repository is the official implementation of [Pisces: A multi-modal data augmentation approach for drug combination synergy prediction](https://www.biorxiv.org/content/10.1101/2022.11.21.517439v1). The code is originally forked from [Fairseq](https://github.com/pytorch/fairseq) and [DVMP](https://github.com/microsoft/DVMP).

## Requirements and Installation
* PyTorch version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

You can build the [Dockerfile](Dockerfile) or use the docker image `teslazhu/pretrainmol36:latest`.

To install the code from source
```
git clone https://github.com/linjc16/Pisces.git

pip install fairseq
pip uninstall -y fairseq 

pip install ninja
python setup.py build_ext --inplace
```

# Getting Started
## Experiments folder
Here we reproduced all three tasks across different settings in our code base, including cell-line-based drug synergy prediction, xenograft-based drug synergy prediction and drug-drug interaction prediction. The mapping between our scripts folder and the particular experiments is as following:
``` 
Pisces/  
├── scripts/  
│   ├── gdsc_trans/ # Vanilla cross validation on GDSC-Combo
│   ├── gdsc_leave_comb/ # Split by combination on GDSC-Combo
│   ├── gdsc_leave_cell/ # Split by cell line on GDSC-Combo
│   ├── xenograft_best_response/ # BestResponse prediction on Xenografts
│   ├── xenograft_days_response/ # Drug combination response prediction across all time points on Xenografts
│   ├── xenograft_extrapolation/ # Response prediction at the last time point on Xenografts
│   ├── drugbank_trans/ # Drug-drug interaction vanilla cross validation on DrugBank
│   ├── drugbank_ind/ # One new drug in each test pair on DrugBank
│   ├── drugbank_unseen/ # Two new drugs in each test pair on DrugBank
│   └── two_sides # Vanilla cross validation on TwoSIDES
└── src/  
```
All the experiments follow the similar pipelines to reproduce the results. Now we take vanilla cross validation on GDSC-Combo as an example to illustrate such a process.
## Data Preprocessing
We evaluate our models on the dataset above. To generate the binary data for `fairseq`, take the `5-fold CV` setting (fold 0) as an example, run
```
python Pisces/scripts/gdsc_trans/data_process/run_process.py

bash Pisces/scripts/gdsc_trans/data_process/run_binarize.sh 0
```

Note that you need to change the file paths accordingly. More original data can be found [here](https://figshare.com/articles/dataset/Pisces_dataset/23272049).

## Training and Test
All training and test scripts can be seen in `Pisces/scripts/gdsc_trans`. For instance,
```
bash Pisces/scripts/gdsc_trans/run.sh 0 0

bash Pisces/scripts/train_trans/inference/inf.sh 0 0
```

## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
