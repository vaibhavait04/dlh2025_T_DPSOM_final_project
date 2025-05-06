# SOM, DSOM, T-DPSOM revisit as a part of DLH 2025 courseware 
## netID: vs39 

## Reference
> Laura Manduchi, Matthias Hüser, Martin Faltys, Julia Vogt, Gunnar Rätsch,and Vincent Fortuin. 2021. T-DPSOM - An Interpretable Clustering Methodfor Unsupervised Learning of Patient Health States. InACM Conference onHealth, Inference, and Learning (ACM CHIL ’21), April 8–10, 2021, VirtualEvent, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3450439.3451872


## Create conda environment 

Create conda environment using requirements file. 
```
conda create --name dlh2025 --file requirements.txt
conda activate dlh2025
```

## Training and Evaluation steps done: 

### Deep Probabilistic SOM

The training script of DPSOM model is `dpsom/DPSOM.py`, the model is defined in `dpsom/DPSOM_model.py`.
Additional changes to DPSOM.py to expedite the runs: `num_epochs`, `decay_steps`, and top 1000 records were used from `data_train`.
To train and test the DPSOM model on the MNIST dataset using default parameters and feed-forward layers:

````python DPSOM.py````

This will train the model and then it will output the clustering performance on test set.  Convolutional layers were not used for this experiment.

Other configuration used for the experiment:
- `validation`: if True it will evaluate the model on validation set (default False).

To reconstruct the centroids of the learned 2D SOM grid into the input space we refer to the Notebook `notebooks/centroids_rec.ipynb`.

### Temporal DPSOM

#### eICU preprocessing pipeline

The major preprocessing steps, which have to be performed sequentially, starting from the raw eICU tables in CSV format, are listed below. The scripts expect the tables to be stored in `data/csv`. Intermediate data is stored in various sub-folders of `data`.

(a) Conversion of raw CSV tables, which can be downloaded from https://eicu-crd.mit.edu/ after access is granted, to HDF versions of the tables. (`eicu_preproc/hdf_convert.py`)

(b) Filtering of ICU stays based on inclusion criteria.  (`eicu_preproc/save_all_pids.py`, `eicu_preproc/filter_patients.py`)

(c) Batching patient IDs for cluster processing (`eicu_preproc/compute_patient_batches.py`)

(d) Selection of variables to include in the multi-variate time series, from the vital signs and lab measurement tables.  (`eicu_preproc/filter_variables.py`)

(e) Conversion of the eICU data to a regular time grid format using forward filling imputation, which can be processed by VarTPSOM.  (`eicu_preproc/timegrid_all_patients.py`, `eicu_preproc/timegrid_one_batch.py`)

(f) Labeling of the time points in the time series with the current/future worse physiology scores as well as dynamic mortality, which are used in the enrichment analyses and data visualizations.  (`eicu_preproc/label_all_patients.py`, `eicu_preproc/label_one_batch.py`)

#### Saving the eICU data-set

Insert the paths of the obtained preprocessed data into the script `eicu_preproc/save_model_inputs.py` and run it.

The script selects the last 72 time-step of each time-series and the following labels: `'full_score_1', 'full_score_6', 'full_score_12', 'full_score_24', 'hospital_discharge_expired_1', 'hospital_discharge_expired_6', 'hospital_discharge_expired_12', 'hospital_discharge_expired_24', 'unit_discharge_expired_1', 'unit_discharge_expired_6', 'unit_discharge_expired_12', 'unit_discharge_expired_24'`
It then saves the dataset in a csv table in `data/eICU_data.csv`.


Note: python conda environment needs to include these basic packages and can have conflict during installation - these might require additional steps, but overall LLM are helpful provide great suggestions on next steps. Packages: `tensorflow tensorflow_probability keras tf-keras torch scikit-learn hd5py pandas matplotlib seaborn jupyter `
