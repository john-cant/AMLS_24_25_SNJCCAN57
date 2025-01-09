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
## 15122024 Extended HyperParameter
## 16122024 Again extended HyperParameter e.g. layers, dropout, filter2
## 16122024 Integrated extended analysis code from Hyper script into library to faciitate sharing
## 20122024 Extended parameter again
## 27122024 Comments and modifications for Task B1 CNN Tune
## 31122024 Extended dataclasses and enhanced hyper analysis in combination with model scripts

#################################################### LIBRARY IMPORTS ##############################
## standard python libraries
import datetime
from dataclasses import dataclass, fields
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
## set up tensorflow
import tensorflow as tf
## MedMNIST specific libraries loading all relevant items (updated 07122024)
import medmnist
from medmnist import INFO ##, info
## sklearn to allow analysis of hyperparameter choices
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

#################################################### SET UP DATACLASSES ##############################
@dataclass
class HyperParameters:
    """ data class to allow storage and passing set of hyperparameters as structure
    """
    learning_rate: float
    kernel_size: int
    num_epochs: int
    optimise: str
    loss: str
    num_filter: int
    strides: int = 1
    padding: str = "valid"
    dropout_rate: float = 0.2
    layers: int = 3
    default_activation: str = "relu"

    def list_parameters(self):
        """ lists all attributes and values in HyperParameters class
        """
        result = ""
        # Loop through attributes and get their values
        for field in fields(HyperParameters):
            attribute_name = field.name
            value = getattr(self, attribute_name)
            result= result+(f"{attribute_name}: {value}"+"\n")
        return result
    
    @classmethod
    def load_excel(cls, file_path: str):
        """
        Loads a single dataclass instance from an Excel file.
        Returns HyperParameter: An instance of the dataclass with values from the Excel file.
        """
        # Read the Excel file into a DataFrame
        df_load = pd.read_excel(file_path)
        # Ensure the file contains at least one row
        if df_load.empty:
            raise ValueError(f"The file {file_path} is empty.")
        # Convert the first row of the DataFrame into a dictionary
        data_dict = df_load.iloc[0].to_dict()
        # Pass the dictionary as keyword arguments to the dataclass constructor
        return cls(**data_dict)
    
    def save_excel(self, file_path: str):
        """
        Saves the dataclass instance to an Excel file.
        """
        # Convert the dataclass to a dictionary
        data = self.__dict__
        # Convert the dictionary into a pandas DataFrame
        df_save = pd.DataFrame([data])  # Wrap in a list to create a single-row DataFrame
        # Save the DataFrame to Excel
        df_save.to_excel(file_path, index=False)

@dataclass
class RunResult:
    """ data class to allow storage and passing of summary run results as structure
    """
    min_loss: float
    max_acc: float
    last_loss: float
    last_acc: float
    min_val_loss: float
    max_val_acc: float
    last_val_loss: float
    last_val_acc: float
    var_loss: float
    var_acc: float

class TqdmEpochProgress(tf.keras.callbacks.Callback):
    """ simple progress bar
    """
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None

    def on_train_begin(self, logs=None):
        """ set up for start of training run
        """
        self.progress_bar = tqdm(total=self.total_epochs, desc="Epoch Progress", unit="epoch")

    def on_epoch_end(self, _, logs=None):
        """ update after each epoch
        """
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs)

    def on_train_end(self, logs=None):
        """ close out at end of training run
        """
        self.progress_bar.close()

#################################################### UTILITY FUNCTIONS ##############################
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
    ## reformat it into a timestamp with year, month, day and time in hours, minutes and seconds
    ## seconds added to avoid overwriting for short hyperparameter selection runs
    return now.strftime("%Y_%m_%d_at_%H%M%S") #timestamp

#################################################### DATA LOADING ##############################

def medMNIST_load(data_name,batch_size,shuffle="N"):
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
        ## updated for more consistent use of data_class
        info          = INFO[data_name]     # info about this dataset
        data_class    = getattr(medmnist, info['python_class'])
        train_dataset = data_class(split='train', download=download)
        test_dataset  = data_class(split='test', download=download)
        val_dataset   = data_class(split='val', download=download)
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
        ## if shuffle equals Y then train dataset is explicitly shuffled
        if shuffle == "Y":
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))\
                                                                .shuffle(len(train_data))\
                                                                .batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))\
                                                                .batch(batch_size)
        test_dataset  = tf.data.Dataset.from_tensor_slices((test_data, test_labels))\
                                                            .batch(batch_size)
        val_dataset   = tf.data.Dataset.from_tensor_slices((val_data, val_labels))\
                                                            .batch(batch_size)
        result_set    = [train_dataset,test_dataset,val_dataset]
    return result_set # set of results

########################################### GRAPHING, SAVING and ANALYSIS ##############################
def graph_and_save(history,summary,parameter,filebase,skip=0):
    """ this version calls the two functions together
       summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary
       parameter is hyperparameter store
       skip allows later start point for graphs (default is 0).
       all data goes to files whatever
    """
    graph(history,summary,parameter,skip)
    ## dump history metrics to excel and model and hyper parameters summary
    ## to text file both with same timestamp in names
    run_summary = history_to_excel(history,
                                   str(summary),
                                   parameter,
                                   filebase)
    print("Files saved:",run_summary[0],run_summary[1])
    return run_summary # [filename_h,filename_s,run_result,parameter] 

def graph(history,summary,parameter,skip=0):
    """summarize history for accuracy
       history is full history of metrics for all epochs
       summary is model summary (not used, but passed for consistency)
       parameter is hyperparameter store
       skip allows later start point for graphs (default is 0).
    """
    ## first load the keys supplied as part of history.
    ## used to dynamically set various graph elements
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
    print("for model\n",str(summary))
    # no return

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
    ## then organise the rest of the dataframe ready for saving
    metrics_df = metrics_df[column_order]
    ## write dataframe to filename formed by appending timestr to filebase
    timestr    = get_timestamp()
    filename_h = filebase+'metrics_'+timestr+'.xlsx'
    metrics_df.to_excel(filename_h,index=False)
    ## now open the summary text file with matching timestamp
    filename_s = filebase+'summary_'+timestr+'.txt'
    ## write the parameter text to the file
    with open(filename_s, "w") as file:
        file.write(parameter.list_parameters()+summary)
    ## now construct the run result structure with calculated metrics
    run_result = RunResult(min_loss      = metrics_df[column_order[1]].min(),
                           max_acc       = metrics_df[column_order[2]].max(),
                           last_loss     = metrics_df[column_order[1]].iloc[-1],
                           last_acc      = metrics_df[column_order[2]].iloc[-1],
                           min_val_loss  = metrics_df[column_order[3]].min(),
                           max_val_acc   = metrics_df[column_order[4]].max(),
                           last_val_loss = metrics_df[column_order[3]].iloc[-1],
                           last_val_acc  = metrics_df[column_order[4]].iloc[-1],
                           var_loss      = metrics_df[column_order[1]].var(),
                           var_acc       = metrics_df[column_order[2]].var())
    ## added parameter to return results
    return [filename_h,filename_s,run_result,parameter] #filenames

def hyper_process(history,_,parameter):
    """ flexibly reads history and writes to dataframe to simplify analysis
        packages a return structure of runresult dataframe and paramter set
        expanded list of parameters that are handled
    """
    keys = list(history.history.keys())
    ## organise the dataframe columns
    column_order = []
    column_order.append('epoch')
    for item in keys:
        column_order.append(item)
    ## convert history which is dictionary structure to a DataFrame
    metrics_df = pd.DataFrame(history.history)
    ## add an epoch column to the dataframe for ease of access
    metrics_df['epoch'] = metrics_df.index + 1
    ## organise the dataframe columns
    metrics_df = metrics_df[column_order]
    run_result = RunResult(min_loss      = metrics_df[column_order[1]].min(),
                           max_acc       = metrics_df[column_order[2]].max(),
                           last_loss     = metrics_df[column_order[1]].iloc[-1],
                           last_acc      = metrics_df[column_order[2]].iloc[-1],
                           min_val_loss  = 99999,
                           max_val_acc   = 0,
                           last_val_loss = 99999,
                           last_val_acc  = 0,
                           var_loss      = metrics_df[column_order[1]].var(),
                           var_acc       = metrics_df[column_order[2]].var())
    ## added parameter to return results
    hyper_history = ["","",run_result,parameter]
    return hyper_history ## ["","",run_result,parameter] to mirror history_to_excel returns

def analyse_run(run_list,selection,filebase):
    """ take run results and analyse
        added filebase param to allow saving of data to file 27122024
    """
    # Extract data into a flat structure
    flat_data = []
    for entry in run_list:
        ## need to flatten structure for both runresult and parameter
        metrics_file, summary_file, result,parameter = entry
        flat_data.append({
            'metrics_file': metrics_file,
            'summary_file': summary_file,
            # RunResult attributes
            'min_loss': result.min_loss,
            'max_acc': result.max_acc,
            'last_loss': result.last_loss,
            'last_acc': result.last_acc,
            'min_val_loss': result.min_val_loss,
            'max_val_acc': result.max_val_acc,
            'last_val_loss': result.last_val_loss,
            'last_val_acc': result.last_val_acc,
            'var_loss': result.var_loss,
            'var_acc': result.var_acc,
            # HyperParameters attributes
            'learning_rate': parameter.learning_rate,
            'kernel_size': parameter.kernel_size,
            'num_epochs': parameter.num_epochs,
            'num_filter': parameter.num_filter,
            'strides': parameter.strides,
            'padding':parameter.padding,
            'dropout_rate': parameter.dropout_rate,
            'layers': parameter.layers,
            'optimise': parameter.optimise,
            'loss': parameter.loss,
            'default_activation': parameter.default_activation,
        })
    # Convert to DataFrame
    run_df = pd.DataFrame(flat_data)
    ## added save to excel 27122024
    timestr    = get_timestamp() #' may pass this in as param to match other filenames
    filename_r = filebase+'run_'+timestr+'.xlsx'
    run_df.to_excel(filename_r,index=False)
    ## Select the run with the smallest min_loss
    min_loss_run = run_df.loc[run_df['min_loss'].idxmin()]
    ## Select the run with the largest max_acc
    max_acc_run = run_df.loc[run_df['max_acc'].idxmax()]
    ## Select the runs that satisfies the selected criteria
    best_run = run_df.loc[(run_df['min_loss'] == run_df['min_loss'].min()) &
                    (run_df['max_acc'] == run_df['max_acc'].max())]
    if len(best_run) == 0:
        print("No single run matches both objectives, so individually")
        print("Run with the smallest min_loss:")
        print(min_loss_run)
        print("\nRun with the largest max_acc:")
        print(max_acc_run)
    ## Select the runs that satisfy further selected criteria
    ## second best run is the one that maximises validation accuracy and
    ## where maximum accuracy is greater or equal to last accuracy, so plateau or increasing
    best_run2 = run_df.loc[(run_df['max_acc'] == run_df['max_acc'].max()) &
                    (run_df['max_acc'] >= run_df['last_acc'])]
    ## third best run is the one that mimimises validation loss and
    ## where maximum accuracy is less than or equal to last loss, so plateau or falling
    best_run3 = run_df.loc[(run_df['min_loss'] == run_df['min_loss'].min()) &
                    (run_df['min_loss'] <= run_df['last_loss'])]
    return run_df,best_run,best_run2,best_run3

def analyse_hyperparameters(run_df):
    """ analyse hyperparameters using several techniques to gauge their impact
    """
    ## Prepare the input analysis data with hyperparameters as features
    ## doesnt support loss or optimise as they are not numeric values (yet)
    X = run_df[['learning_rate', 'num_epochs', 'num_filter','strides','layers',\
                'dropout_rate','kernel_size']]  # Hyperparameters
    y = run_df['max_acc']  ## Metric to predict should this be accuracy or loss?
    ## Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ## Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    ## Evaluate predicted versus actual
    y_pred = model.predict(X_test)
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    ## calculate coefficients for each to understand impact
    coef = pd.DataFrame({'Hyperparameter': X.columns, 'Coefficient': model.coef_})
    ## Fit Random Forest Regressor for max_acc
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, run_df['max_acc'])
    ## calculate feature importance
    feature_importance = pd.DataFrame({
        'Hyperparameter': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return feature_importance,coef
