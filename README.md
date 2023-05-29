# PBXAI Implementation
This project includes all source codes to reproduce the experimental result of MIMIC-III dataset which we report in the article entitled "Interpretable Disease Prediction based on Reinforcement Path Reasoning over Knowledge Graph".

Note, because the PLAGH dataset is a private dataset, we don't release the related source code in this project. If a reader wants to reproduce the experimental result of PLAGH dataset, he/she needs to get the permission from the big medical data center of PLA General Hospital in advance (please contact the corresponding author of our article), and we will send the codes later.

In the following subsection, we will introduce the detail of reproducing.

We have tested the correctness of the project at Windows 10 and Ubuntu 18.04.

## Prerequisite
### Data
We assume you are already able to access the [MIMIC-III (V1.4)](https://mimic.physionet.org/about/mimic/), which is a publicily available electronic health record (EHR) database that contains over 40 thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.  
If you are not able to access MIMIC-III, please complete a required training course and then submit an application at [request link](https://mimic.physionet.org/gettingstarted/access/). Once the application is approved, you can download the MIMIC-III freely.
Please ensure your computer has enough storage space because MIMIC-III is a large dataset (45 GiB is required in this study)

### Environment
Please install Python 3.9 environment, as well as pytorch, matplotlib, numpy, pandas, scikit-learn, and scipy packages in advance.

## Step 1 Uncompress Data
Please clone the project, and then uncompress the entire MIMIC-III dataset into:  
```
/resource/raw_data/mimic/
```
The files in the folder are like:  
```
/resource/raw_data/mimic/  
    ADMISSIONS.csv...
    CALLOUT.csv...
    ......
    TRANSFERS.csv
```

  
## Step 2 Build the preprocessed dataset from the raw data
Please run the following scripts successively.
```
    1.   src/data_preprocess/mimic_patient_feature_generator.py
    2.   src/data_preprocess/mimic_feature_selection.py
    3.   src/data_preprocess/mimic_visit_selection_and_reorganize.py
    4.   src/data_preprocess/mimic_distribution_convert_and_impute.py
    5.   src/data_preprocess/mimic_data_split.py
```
These five scripts are responsible to reconstruct the dataset to a structured format and extract the label. As the MIMIC-III is a large dataset, the first script needs a long time to be executed (it takes about 1 hour in my computer). 

Once the scripts are executed successfully, we can find a file named 'mimic_imputed_data.csv' in the /resource/preprocessed_data folder, and the split data in the mimic_five_part_five_fold folder and the mimic_two_part_five_fold folder.
  
## Step 3 Learn representations of patient and medical entity
Please run the following scripts successively.
```
    1.   src/representation_learning/concept_learning.py
    2.   src/representation_learning/representation_learning.py
```
These two scripts are responsible to learn the representations of patients and medical entities, as we described in the subsection-III.C of our article.

Once the scripts are executed successfully, we can find files are created in the resource/representation folder

## Step 4 Run the model
Please run the script:
```
    1.   src/model/train_agent.py
```

Once the script is executed successfully, we can find files in the resource folder, which is the result we reported in our article
