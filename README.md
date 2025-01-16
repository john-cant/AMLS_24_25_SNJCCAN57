# **AMLS Assessment:Project Title and Description**
This project is the implementation, analysis and results for two supervised machine learning categorization problems, Tasks A and B, as required by the AMLS ELEC0134 24/25 Assignment. 
-	Task A is a binary classification task using BreastMNIST publicly available data set with the objective of correctly classifying images into Benign and Malignant.
-	Task B is a multi-classification task using BloodMNIST data set with the objective of classify-ing images into one of eight different blood types.
The key scripts to run the various hyperparameter selections, training and testing were implemented in Jupyter notebooks running on a specific Anaconda configured environment. These reference both external and locally developed Python library - AMLS_common.py – the later to reduce the need for duplicated code and to improve the relevance, efficiency and readability of the scripts.

## **Installation Instructions:**
This project is intended to be easy to run and need the minimum of installation. The files can be run from a copy of the GitHub structure.
All the Python packages required are listed in the Packages Required section below. There is a single main.py file that will run the main scripts.
In addition a copy of a local code library, - developed for these specific tasks and includes all references to MedMNIST specific library - called AMLS_common.py needs to be included in each of the folders A & B.
Each folder A and B should have a metrics folder which will be used to store results files of various types. If this does not exist then main will create it.


## **Usage Examples:**
python main.py 
Will run the three main Tasks Jupyter Notebook files -Task A SVM, Task A CNN and Task B CNN


## **Features:**
There are seven main scripts/Jupyter notebooks:
- Task_A_1_SVM_Base: the SVM models for Task A
- Task_A_2_CNN_Base: the simple Base CNN model that shows the Task A data loads and the model runs
- Task_A_3_CNN_Hyper: supports the HyperParameter set selection for the Task A CNN using the Training and Validation datasets
- Task_A_4_CNN_Tune: takes the optimised HyperParameters and runs the model against the Test Dataset, outputting results
- Task_B_1_CNN_Base: the simple Base CNN model that shows the Task B data loads and the model runs
- Task_B_2_CNN_Hyper: supports the HyperParameter set selection for the Task B CNN using the Training and Validation datasets
- Task_B_3_CNN_Tune: takes the optimised HyperParameters and runs the model against the Test Dataset, outputting results

## **Packages Required:**
- io
- os
- time
- datetime         
- numpy
- matplotlib.pyplot
- tensorflow
- tensorflow.keras.models 
- tensorflow.keras.layers
- tensorflow.keras.optimizers
- tensorflow.keras.losses
- dataclasses
- pandas
- tqdm
- medmnist
- sklearn.linear_model
- sklearn.model_selection
- sklearn.metrics
- sklearn.ensemble
- sklearn.svm
- nbconvert

## **Function details within AMLS Common**
A core design principle is to keep as much of the com-mon functionality in this single Python library to reduce length, duplication, and improve efficiency and readability of all model scripts. The following describes the core components in this library.

_Data Classes_
-	HyperParameter: data class to allow storage and passing of set of hyperparameters as a structure. Contains possible defaults, and also specific list, load and save Excel subfunctions
-	RunResult: to allow storage and passing of summary model run results

_medMNIST Load_
There are two datasets used, one per Task. These are supplied as part of the MedMNIST standardised biomedical datasets which are supported for biomedical, ML and computer vision modelling in various resolutions and for-mats. The specifics for each task are:
-	Task A BreastMNIST: the BreastMNIST data set consists of 28 x 28 resolution images and then a set of binary labels (Benign/Malignant). It contains 546 images for Training, 78 for Validation and 156 for Testing. The load function delivers these as three distinct Tensorflow datasets using a standard batch size.
-	Task B BloodMNIST: the BloodMNIST data set 28 x 28 x 3 resolution. The additional dimension provides colour information. There is also a set of labels defining which of the eight blood types the image relates to. It contains 11,959 images for Training, 1,715 for Validation and 3,421 for Test-ing. 
The load function delivers these as three distinct Tensorflow datasets using a standard batch size. It also does some basic normalising of the data, including confirming the label datatype. The function also has the option to shuffle the Training dataset dependent on a parameter. As part of the testing of this function, I compared and verified the data sourced via both methods in the AMLS library, i.e. tensorflow and numpy with a more direct externally written torch routine to ensure it was loaded correctly and consistently.

_Utility Functions_
-	Dataset to NumPy: converts a Tensorflow dataset into a NumPy arrays
-	Get TimeStamp: Gets date/timestamp to allow traceable filenames for run metrics/parameter files
-	TDQM Epoch Progress: to allow displaying of progress metric to track longer running epochs
-	StopOverfittingCallback: implements overfitting reduction function as defined in 4.1.4

_Analysis Functions_
These support rich analysis plus graphing and results storage in structured files.
-	Graph and Save: Calls Graph and History to Excel
-	Graph: outputs multi-quadrant graph for accuracy and loss for given run history. The skip parameter can define the first epoch to be plotted. This is useful where much of the change occurs in the first few epochs suppressing later differences due to scaling.
-	Graph Compare: allows display of two model runs from metrics files on a single plot.
-	History to Excel: flexibly writes a run history set of accuracy and loss values to an excel file and a linked text file with model structure and parame-ters.
-	Hyper Process: flexibly reads run history and writes to run result dataframe to simplify analysis.
-	Analyse Run: saves hyper run output and then se-lects best set using combination of criteria.
-	Analyse HyperParameters: uses several techniques, including linear and random forest regression to gauge their impact.
