import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path, output_path=None, timeout=36000):
    """
    Executes a Jupyter Notebook and optionally saves the output.
    :param notebook_path: Path to the notebook file to execute.
    :param output_path: Path to save the executed notebook (optional).
    :param timeout: Execution timeout for each notebook cell (default: 600 seconds).
    """
    ## Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
    ## Set up the notebook executor make sure it is the correct kernel name
    executor = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    try:
        ## Execute the notebook
        executor.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})

        # #Save the executed notebook if an output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                nbformat.write(notebook, output_file)       
        print(f"Executed notebook: {notebook_path}")
    except Exception as e:
        ## catch errors
        print(f"Failed to execute {notebook_path}: {e}")

def folder_safe(directory_path):
    """ Check if the directory exists
        and if not then create it
    """
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def main():
    """ main script execution
    """
    ## Define the paths to the specific notebooks
    ## Should be set to A and B for official running
    ## Can be changed for development test
    path_a = "A"
    path_b = "B"
    folder_safe(path_a+"/metrics")
    folder_safe(path_b+"/metrics")
    notebook_a_svm_path = path_a+"/TASK_A_1_SVM_Base.ipynb"
    notebook_a_cnn_path = path_a+"/TASK_A_4_CNN_Tune.ipynb"
    notebook_b_cnn_path = path_b+"/TASK_B_3_CNN_Tune.ipynb"
    ## Define where to save the executed notebooks (optional)
    executed_a_svm_path = path_a+"/executed_notebook_a_SVM.ipynb"
    executed_a_cnn_path = path_a+"/executed_notebook_a_CNN.ipynb"
    executed_b_cnn_path = path_b+"/executed_notebook_b_CNN.ipynb"
    ## Execute all notebooks
    run_notebook(notebook_a_svm_path, executed_a_svm_path)
    run_notebook(notebook_a_cnn_path, executed_a_cnn_path)
    run_notebook(notebook_b_cnn_path, executed_b_cnn_path)


if __name__ == "__main__":
    main()
