
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from .utils import create_directories_from_paths
from .utils import print_and_convert_size
from .utils import get_or_print_filesize
from .utils import deep_getsizeof
from .utils import get_filesize


def save_filepath_config(FPATHS, overwrite=True, output_fpath = 'config/filepaths.json',
                        verbose=False):
    ## Save the filepaths 
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    if os.path.exists(output_fpath) & (overwrite==False):
        raise Exception(f"- {output_fpath} already exists and overwrite is set to False.")

    with open(output_fpath, 'w') as f:
        json.dump(FPATHS, f)

    print(f"[i] Filepath json saved as {output_fpath} ")
    if verbose:
        pprint(FPATHS)
        
    return FPATHS

import os





def load_filepath_config(config_fpath = 'config/filepaths.json', verbose=True):
    with open (config_fpath) as f:
        FPATHS = json.load(f)
    
    if verbose == True:
        print(f"- Filepaths loaded successfully:")
        pprint(FPATHS)
        
    return FPATHS




def save_ml_model_and_data(model, fpath_model, train_data = [], fpath_train_data=None, 
                           test_data = [], fpath_test_data=None, verbose=True):
    """Save a machine learning model with joblib, optionally, also save the training and test data.

    Args:
        model (model, optional): Serializable model to save to joblib.
        fpath_model (str, optional): filepath for the model.joblib 
        train_data (list, optional): [X_train, y_train] . Defaults to [].
        fpath_train_data (list, optional): filepath for traim data joblib. Defaults to None.
        test_data (list, optional): [X_test, y_test] . Defaults to [].
        fpath_test_data (str, optional): filepath for test data joblib.. Defaults to None.
        verbose (bool, optional): Whether to print the file size or not. Defaults to True.
    """
    import joblib
    # Save Model
    joblib.dump(model, fpath_model)
    
    if verbose:      
        size = get_or_print_filesize(fpath_model, unit="MB",print_or_return="return")
        print(f"\n- Model saved to {fpath_model} (filesize: {size}")
        

    ## Save training data if provided
    if (len(train_data) > 0) | (fpath_train_data is not None):
        joblib.dump(train_data, fpath_train_data)
        if verbose:      
            size = get_or_print_filesize(fpath_train_data, unit="MB",print_or_return="return")
            print(f"- Train data saved to {fpath_train_data} (filesize: {size})")

    ## Save test data if provided
    if (len(test_data) > 0) | (fpath_test_data is not None):
        joblib.dump(test_data, fpath_test_data)
        if verbose:      
            size = get_or_print_filesize(fpath_test_data, unit="MB",print_or_return="return")
            print(f"- Test data saved to {fpath_test_data} (filesize: {size}")
        


def load_model_results(fpath_dict=None,report_fname=None, conf_mat_fname=None):
    
    if fpath_dict is not None:
        if ( 'confusion_matrix' not in fpath_dict) | ('classification_report' not in fpath_dict):
            raise Exception("If using fpath_dict, must contain keys: 'confusion_matrix','classification_report'")
            
        else:
            conf_mat_fname = fpath_dict['confusion_matrix']
            report_fname =  fpath_dict['classification_report']
            
    from PIL import Image
    conf_mat = Image.open(conf_mat_fname)

    with open(report_fname) as f:
        report = f.read()
    
    return report, conf_mat
    
