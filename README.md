# 2023_URTmetaanalysis
This repository contains the analysis code for the publication, "Meta-analysis of the human upper respiratory tract reveals robust taxonomic associations with health and disease" by Quinn-Bohmann et al., 2024. 
_____

## Data Availability
All raw data files used in this analysis are available on NCBI SRA, with one exception. Accession codes for these data can be found in Additional Data File 1, Table 1. 

Intermediate data files created using QIIME2 can be found on Zenodo, under the following DOI: 10.5281/zenodo.11038446. These data are required for the analysis contained herein. 

To properly conduct the analyses aftering forking this repository, please download the intermediate data from Zenodo and add to the "data" directory. All data are separated into nasopharyngeal (NP) and oropharyngeal (OP) samples. Ensure that all NP samples, including NP_manifest.csv, and all files in metadata_NP.zip and qiime_NP.zip are deposited in the NP directory, and the same for OP samples. This will help ensure files paths in the notebooks are accurate. 

_____

## Analysis
To replicate the analysis from the paper, please start with the notebook entitled merged_table.ipynb. This will constructed a table of merged reads, used in all subsequent analyses. Other notebooks are listed below:

1. classification.ipynb examines success of taxonomic classification of reads, and constructs Additional File 2, Fig. S2. 
2. diversity.ipynb conducts analysis of alpha and beta diversity, and constructs Fig. 1 and Fig. 2.
3. covariate_analysis.ipynb conducts analysis of taxonomic patterns between geographic location, age and sex in healthy samples, and constructs Fig. 3.
4. auroc.ipynb builds random forest classifiers and judges them using AUROC analysis, and constructs Fig. 4.
5. logistic_regression.ipynb conducts per-study logistic regression of cases vs. controls, and constructs Fig. 5.  
