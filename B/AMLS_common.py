## Common code to support all tasks of AMLS Assigmnent
## including the loading the MedMNIST data files into tensorflow format
## loading of hyperparameters into single data class structure
## and handling of NN history results: graphing and storing results in files
## Used across both assignment tasks
## import common libraries
## Revision History
## 02122024 Tidy functions & comments (including pylint run)
## 02122024 Split graph and save functions to allow graphing without saving for heavy testing
## 02122024 Update graph function to allow plot start at skip to ease analysis
## 07122024 Updated for more stable use of medmnist library and associated functions
## 09122024 Enhanced dataclass with defaults and list function for saving to file
## 09122024 Add tqdm custom callback 
## 10122024 Bug fix for timestamp - use minutes not seconds

import datetime
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
## import tensorflow
import tensorflow as tf
## MedMNIST specific libraries loading all relevant items (updated 07122024)
import medmnist
from medmnist import INFO
from medmnist.dataset import BreastMNIST, BloodMNIST

@dataclass
class HyperParameters:
    """ data class to allow storage and passing of hyperparameters as structure
    """
    learning_rate:float
    batch_size: int
    num_epochs: int
    optimise: str
    loss:str
    default_activation: str

    def list_parameters(self):
        result = ""
        # Loop through attributes and get their values
        for field in fields(HyperParameters):
            attribute_name = field.name
            value = getattr(self, attribute_name)
            result= result+(f"{attribute_name}: {value}"+"\n")
        return result

class TqdmEpochProgress(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None

    def on_train_begin(self, logs=None):
        self.progress_bar = tqdm(total=self.total_epochs, desc="Epoch Progress", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs)

    def on_train_end(self, logs=None):
        self.progress_bar.close()

def medMNIST_load(data_name,batch_size):
    """ reads in data_name and batch_size and returns set of tensor flow datasets
    """
    ## first set up key variables
    download   = True                  ## set to True for all datasets
    result_set = []                    ## structure so all datasets can be returned in one go
    if data_name not in ['breastmnist','bloodmnist']:
        ## for this assessment we only support these two data names
        print("ERROR: Unsupported data_name parameter provided - returning empty result_set")
    else:
        ## Load raw training, test and validation datasets
        ## Make explicit references to one of the two supported sets
        ## updated for more consistent use of DataClass
        info          = INFO[data_name]     # info about this dataset
        DataClass     = getattr(medmnist, info['python_class'])
        train_dataset = DataClass(split='train', download=download)
        test_dataset  = DataClass(split='test', download=download)
        val_dataset   = DataClass(split='val', download=download)
        ## Now access data and labels from loaded datasets
        train_data    = train_dataset.imgs
        train_labels  = train_dataset.labels
        test_data     = test_dataset.imgs
        test_labels   = test_dataset.labels
        val_data      = val_dataset.imgs
        val_labels    = val_dataset.labels
        ## Normalize data including label datatype
        train_data    = train_data / 255.0
        test_data     = test_data / 255.0
        val_data      = val_data / 255.0
        train_labels  = train_labels.astype(np.float32)
        test_labels   = test_labels.astype(np.float32)
        val_labels    = val_labels.astype(np.float32)
        ## Reshape images to fit TensorFlow requirements - if needed
        if data_name == 'breastmnist':
            train_data    = np.expand_dims(train_data, axis=-1)  # Add channel dimension
            test_data     = np.expand_dims(test_data, axis=-1)   # Add channel dimension
            val_data      = np.expand_dims(val_data, axis=-1)    # Add channel dimension
        ## Create TensorFlow Datasets and bundle to return as single result set
        ## train dataset is explicitly shuffled
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))\
                                                            .shuffle(len(train_data))\
                                                            .batch(batch_size)
        test_dataset  = tf.data.Dataset.from_tensor_slices((test_data, test_labels))\
                                                            .batch(batch_size)
        val_dataset   = tf.data.Dataset.from_tensor_slices((val_data, val_labels))\
                                                            .batch(batch_size)
        result_set    = [train_dataset,test_dataset,val_dataset]
    return result_set # set of results

def dataset_to_numpy(dataset):
    """ change from loaded dataset to numpy arrays
        Used to change BreastMNIST dataset for SVM analysis
        In this way we only need one data loader for all analysis types
    """
    ## set up the interim structures to allow concatenation across batches
    x_list = []
    y_list = []
    ## loop through dataset
    for batch in dataset:
        ## Unpack the batch into features (x) and labels (y)
        x_batch, y_batch = batch
        ## Append the batches to the lists
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    ## Concatenate all batches into single NumPy arrays, one for x and one for y
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y #x and y as numpy arrays

def get_timestamp():
    """ Gets timestamp. NB specifically compatible with inclusion in filenames
    """
    ## get current datetime
    now = datetime.datetime.now()
    ## reformat it into a timestamp with year, month, day and time in hours and minutes
    return now.strftime("%Y_%m_%d_at_%H%M") #timestamp

def graph_and_save(history,summary,parameter,filebase,skip=0):
    """ this version calls the two functions together
       summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary
       parameter is hyperparameter store
    """
    keys = list(history.history.keys())
    graph(history,summary,parameter,skip)
    for item in keys:
        print("Item:",item)
    ## dump history metrics to excel and model and hyper parameters summary
    ## to text file both with same timestamp in names
    print("Files saved:",history_to_excel(history,
                                          str(summary),
                                          parameter,
                                          filebase))

def graph(history,summary,parameter,skip=0):
    """summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary (not used, but passed for consistency)
       parameter is hyperparameter store
       skip allows later start point for graphs (default is 0). 
       All data goes to files whatever
    """
    ## first load the keys supplied as part of history. These are then used to 
    ## dynamically set various graph elements
    keys = list(history.history.keys())
    ## work out if there is a single or double graph lines
    graph_type = "two"
    if len(keys) == 2:
        graph_type = "obe"
    ## set the epochs range for use in plot
    epochs = range(1,len(history.history['loss'])+1)
    ## initalise the plot size
    plt.figure(figsize=(12, 5))
    ## set up the first subplot
    plt.subplot(1, 2, 1)
    ## set the first line based on the specific accuracy key supplied
    plt.plot(epochs[skip:],history.history[keys[1]][skip:])
    if graph_type == "two":
        plt.plot(epochs[skip:],history.history[keys[3]][skip:])
    plt.title('model accuracy [lr='+str(parameter.learning_rate)+']')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    ## set the legend depending on whether the graph has one or two lines
    if graph_type == "two":
        plt.legend(['train','val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    ## now set up the second subplot alongside the first
    plt.subplot(1, 2, 2)
    ## summarize history for loss based on supplid loss key
    plt.plot(epochs[skip:],history.history[keys[0]][skip:])
    ## set the legend depending on whether the graph has one or two lines
    if graph_type == "two":
        plt.plot(epochs[skip:],history.history[keys[2]][skip:])
    plt.title('model loss [lr='+str(parameter.learning_rate)+']')
    plt.ylabel(keys[0])
    plt.xlabel('epoch')
    if graph_type == "two":
        plt.legend(['train','val'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    plt.show()

def history_to_excel(history,summary,parameter,filebase):
    """ puts history metrics into unique excel file
        expanded list of parameters that are handled
        further version could take all of the paramter entries and autoadd to file
    """
    keys = list(history.history.keys())
    ## check to see whether val_ variants are provided
    column_order = []
    column_order.append('epoch')
    for item in keys:
        column_order.append(item)
    ## convert history which is dictionary structure to a DataFrame
    metrics_df = pd.DataFrame(history.history)
    ## add an epoch column to the dataframe for ease of access
    metrics_df['epoch'] = metrics_df.index + 1
    metrics_df = metrics_df[column_order]
    ## write dataframe to filename formed by appending timestr to filenamebase
    timestr    = get_timestamp()
    filename_h = filebase+'metrics_'+timestr+'.xlsx'
    metrics_df.to_excel(filename_h,index=False)
    ## now open the summary text file with matching timestamp
    filename_s = filebase+'summary_'+timestr+'.txt'
    ## write the parameter text to the file
    with open(filename_s, "w") as file:
        file.write(parameter.list_parameters()+summary)
    return [filename_h,filename_s] #filenames
