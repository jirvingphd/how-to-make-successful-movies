from .evaluate import  get_true_pred_labels, convert_y_to_sklearn_classes
from sklearn import metrics
from IPython.display import display
import matplotlib.pyplot as plt


def save_classification_metrics(
    y_true,
    y_pred,
    label="",
    output_dict=False,
    figsize=(8, 4),
    normalize="true",
    cmap="Blues",
    colorbar=False,
    values_format=".2f",
    # New Args:
    save_results=False,
    target_names=None,
    report_fname = None,#"Models/results/report.txt",
    conf_mat_fname = None,#"Models/results",
    results_folder="Models/results/",
    model_prefix="ml",
    verbose=True,
    savefig_kws={},
):
    """Classification metrics function from Advanced Machine Learning
    - Saves the classification report and figure to file for easy use"""
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    # Get the classification report
    report = classification_report(
        y_true,
        y_pred,target_names=target_names,
    )
    ## Print header and report
    header = "-" * 70
    # print(header, f" Classification Metrics: {label}", header, sep='\n')
    # print(report)
    final_report = (
        header + "\n" + f" Classification Metrics:    {label}" "\n"+ header + "\n" + report
    )
    print(final_report)

    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    # create a confusion matrix with the test data
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize=normalize,
        cmap=cmap,
        values_format=values_format,
        colorbar=colorbar,
        display_labels=target_names,
        ax=axes[0],
    )
    axes[0].set_title("Normalized Confusion Matrix")

    # create a confusion matrix  of raw counts
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize=None,
        cmap="gist_gray_r",
        values_format="d",
        colorbar=colorbar,
        display_labels=target_names,
        ax=axes[1],
    )
    axes[1].set_title("Raw Counts")

    
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()

    ## New Code For Saving Results
    if save_results == True:
        import os
        # Create the results foder
        os.makedirs(results_folder, exist_ok=True)

        # ## Save classification report
        if report_fname is None:
            report_fname = results_folder + f"{model_prefix}-class-report-{label}.txt"
        if conf_mat_fname is None:
            conf_mat_fname =  results_folder + f"{model_prefix}-conf-mat-{label}.png"
        
        with open(report_fname, "w") as f:
            f.write(final_report)

        if verbose:
            print(f"- Classification Report saved as {report_fname}")

        ## Save figure
        fig.savefig(
            conf_mat_fname, transparent=False, bbox_inches="tight", **savefig_kws
        )
        if verbose:
            print(f"- Confusion Matrix saved as {conf_mat_fname}")

        
        ## Save File Info:
        fpaths={'classification_report':report_fname, 
                 'confusion_matrix': conf_mat_fname}

        return fpaths
        
    # Return dictionary of classification_report
    if output_dict == True:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict 


        
def evaluate_save_classification(model, X_train=None, y_train=None, X_test=None, y_test=None,
                            figsize=(6,4), normalize='true', output_dict = False,
                            cmap_train='Blues', cmap_test="Reds",colorbar=False,label='',
                            # New Args - Saving Results
                            target_names=None, save_results=False,  verbose=True, report_label=None,
                            results_folder="Models/results/", model_prefix="ml", savefig_kws={},
                            # New Args - Saving Model
                            save_model=False, save_data=False,
                            FPATHS = None
                           ):
    """Updated Verson of Intro to ML's evaluate_classification function"""

    ## Adding a Print Header
    print("\n"+'='*70)
    print(f'- Evaluating Model:    {label}')
    print('='*70)

    ## Changing value of output dict if saving results
    if save_results == True:
        output_dict=False

    if ( X_train is not None) & (y_train is not None):
        
        # make the final label names for classification report's header
        if report_label is not None:
            report_label_ = report_label + " - Training Data"
        else: 
            report_label_ = 'Training Data'

        ## NEW: GET FILENAME FROM FPATHS
        if FPATHS is not None:
            if 'ml' in FPATHS['models']:
                report_fname = FPATHS['ml'] 
        
        # Get predictions for training data
        y_train_pred = model.predict(X_train)
        
        # Call the helper function to obtain regression metrics for training data
        fpaths_or_results_train = save_classification_metrics(
            y_train,
            y_train_pred,  
            output_dict=output_dict,
            figsize=figsize,
            colorbar=colorbar,
            cmap=cmap_train,
            target_names=target_names,
            save_results=save_results,
            verbose=verbose,
            label=report_label_,
            results_folder=results_folder,
            model_prefix=model_prefix,
            savefig_kws=savefig_kws,
        )

        print()

    if ( X_test is not None) & (y_test is not None):
        # make the final label names for classification report's header
        if report_label is not None:
            report_label_ = report_label + " - Test Data"
        else: 
            report_label_ = 'Test Data'

        # Get predictions for test data
        y_test_pred = model.predict(X_test)
        
       # Call the helper function to obtain regression metrics for training data
        fpaths_or_results_test = save_classification_metrics(
            y_test,
            y_test_pred,  
            output_dict=output_dict,
            figsize=figsize,
            colorbar=colorbar,
            cmap=cmap_test,
            label=report_label_,
            target_names=target_names,
            save_results=save_results,
            verbose=verbose,
            results_folder=results_folder,
            model_prefix=model_prefix,
            savefig_kws=savefig_kws,
        )

    # Save a joblib file
    if save_model == True:
        import joblib
        to_save = {'model':model}
    
        if save_data == True:
            vars = {'X_train':X_train,'y_train': y_train,'X_train': X_train,'y_train': y_train}
            for name, var in vars.items():
                if var is not None:
                    to_save[name] = var

        # Save joblib
        fpath_joblib = results_folder+f"model-{model_prefix}.joblib"
        joblib.dump(to_save, fpath_joblib)
    
            
        
        
    ## If either output_dict or save_results
    if (save_results==True) | (output_dict==True):
        # Store results in a dict if ouput_frame or save_results is True
        results_dict = {'train':fpaths_or_results_train,
                    'test': fpaths_or_results_test}
        if save_model == True:
            results_dict['model-joblib'] = fpath_joblib
        return results_dict
    


def evaluate_classification_network(
    model,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    history=None,
    history_figsize=(6, 6),
    figsize=(6, 4),
    normalize="true",
    output_dict=False,
    cmap_train="Blues",
    cmap_test="Reds",
    values_format=".2f",
    colorbar=False,
    # return_results=False,
    label="",
    # New Args
    target_names=None,
    save_results=False,
    verbose=True,
    report_label=None,
    results_folder="Models/results/",
    model_prefix="nn",
    savefig_kws={},
    save_model=False, 
    save_data=False,
    model_save_fmt='tf',
    model_save_kws = {}
):
    """Evaluates a neural network classification task using either
    separate X and y arrays or a tensorflow Dataset

    Data Args:
        X_train (array, or Dataset)
        y_train (array, or None if using a Dataset
        X_test (array, or Dataset)
        y_test (array, or None if using a Dataset)
        history (history object)
    """
    import os
    filepaths = {}
    # Plot history, if provided
    if history is not None:
        fig = plot_history(history, figsize=history_figsize)

        ## New Code For Saving Results
        if save_results == True:
            # Create the results foder
            os.makedirs(results_folder, exist_ok=True)

            ## Save classification report
            history_fname = results_folder + f"{model_prefix}-history.png"
            filepaths['history'] = history_fname
            ## Save figure
            fig.savefig(
                history_fname, transparent=False, bbox_inches="tight", **savefig_kws
            )
            if verbose:
                print(f"- Model History saved as {history_fname}")

    ## Adding a Print Header
    print("\n" + "=" * 70)
    print(f"- Evaluating Network:    {label}")
    print("=" * 70)

    ## Changing value of output dict if saving results
    if save_results == True:
        output_dict = False

    ## TRAINING DATA EVALUATION
    # check if X_train was provided
    if X_train is not None:
        # make the final label names for classification report's header
        if report_label is not None:
            report_label_ = report_label + " - Training Data"
        else:
            report_label_ = "Training Data"

        ## Check if X_train is a dataset
        if hasattr(X_train, "map"):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_train, y_train_pred = get_true_pred_labels(model, X_train)
        else:
            # Get predictions for training data
            y_train_pred = model.predict(X_train)

        ## Pass both y-vars through helper compatibility function
        y_train = convert_y_to_sklearn_classes(y_train)
        y_train_pred = convert_y_to_sklearn_classes(y_train_pred)

        # Call the helper function to obtain regression metrics for training data
        fpaths_or_results_train = save_classification_metrics(
            y_train,
            y_train_pred,
            output_dict=output_dict,
            figsize=figsize,
            colorbar=colorbar,
            cmap=cmap_train,
            values_format=values_format,
            label=report_label_,
            target_names=target_names,
            save_results=save_results,
            verbose=verbose,
            results_folder=results_folder,
            model_prefix=model_prefix,
            savefig_kws=savefig_kws,
        )

        ## Run model.evaluate
        print("\n- Evaluating Training Data:")
        print(model.evaluate(X_train, return_dict=True))

    # If no X_train, then save empty list for results_train
    else:
        fpaths_or_results_train = []

    ## TEST DATA EVALUATION
    # check if X_test was provided
    if X_test is not None:
        # make the final label names for classification report's header
        if report_label is not None:
            report_label_ = report_label + " - Test Data"
        else:
            report_label_ = "Test Data"

        ## Check if X_train is a dataset
        if hasattr(X_test, "map"):
            # If it IS a Datset:
            # extract y_train and y_train_pred with helper function
            y_test, y_test_pred = get_true_pred_labels(model, X_test)
        else:
            # Get predictions for training data
            y_test_pred = model.predict(X_test)

        ## Pass both y-vars through helper compatibility function
        y_test = convert_y_to_sklearn_classes(y_test)
        y_test_pred = convert_y_to_sklearn_classes(y_test_pred)

        # Call the helper function to obtain regression metrics for training data
        fpaths_or_results_test = save_classification_metrics(
            y_test,
            y_test_pred,
            output_dict=output_dict,
            figsize=figsize,
            colorbar=colorbar,
            cmap=cmap_test,
            label=report_label_,
            target_names=target_names,
            save_results=save_results,
            verbose=verbose,
            results_folder=results_folder,
            model_prefix=model_prefix,
            savefig_kws=savefig_kws,
        )

        ## Run model.evaluate
        print("\n- Evaluating Test Data:")
        print(model.evaluate(X_test, return_dict=True))
    else:
        fpaths_or_results_test = []


    if save_model:
        model_fpath = results_folder+f"model-{model_prefix}" 
        model.save(model_fpath, **model_save_kws ,save_format=model_save_fmt)

        if save_data:
            # Saving Tensorflow dataset to tfrecord
            if X_test is not None:
                fname_test_ds =results_folder+f"model-{model_prefix}-test-ds" # test_data_folder+"test-ds"#.tfrecord"
                X_test.save(path=fname_test_ds,)
            else: 
                raise Exception("[!] save_data=True, but X_test = None!")
                
            

    
    ## If either output_dict or save_results
    if (save_results == True) | (output_dict == True):
        # Store results in a dict if ouput_frame or save_results is True
        results_dict = {**filepaths,
            "train": fpaths_or_results_train,
            "test": fpaths_or_results_test,
        }

        if save_model == True:
            results_dict['model'] = model_fpath

            if save_data == True:
                results_dict['test-ds'] = fname_test_ds
            
        return results_dict





## Update to add option to save  (
# or just return the plot and have the evaluate_classification_network function
# do the saving)
def plot_history(history, figsize=(6, 8)):
    import matplotlib.pyplot as plt
    import numpy as np
    # Get a unique list of metrics
    all_metrics = np.unique([k.replace("val_", "") for k in history.history.keys()])

    # Plot each metric
    n_plots = len(all_metrics)
    fig, axes = plt.subplots(nrows=n_plots, figsize=figsize)
    axes = axes.flatten()

    # Loop through metric names add get an index for the axes
    for i, metric in enumerate(all_metrics):
        # Get the epochs and metric values
        epochs = history.epoch
        score = history.history[metric]

        # Plot the training results
        axes[i].plot(epochs, score, label=metric, marker=".")
        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}", marker=".")
        except:
            pass

        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch", ylabel=metric)

    # Adjust subplots and show
    fig.tight_layout()
    plt.show()
    return fig



def evaluate_classification_binary(model, X_train=None,y_train=None,X_test=None,y_test=None,
                            normalize='true',cmap='Blues', label= ': Classification', figsize=(10,5)):
    """Evaluates a classification model using the training data, test data, or both. 

    Args:
        model (Estimator): a fit classification model
        X_train (Frame, optional): X_train data. Defaults to None.
        y_train (Series, optional): y_train data. Defaults to None.
        X_test (_type_, optional): X_test data. Defaults to None.
        y_test (_type_, optional): y_test data. Defaults to None.
        normalize (str, optional): noramlize arg for ConfusionMatrixDisplay. Defaults to 'true'.
        cmap (str, optional): cmap arg for ConfusionMatrixDisplay. Defaults to 'Blues'.
        label (str, optional): label for report header. Defaults to ': Classification'.
        figsize (tuple, optional): figsize for confusion matrix/roc curve subplots. Defaults to (10,5).

    Raises:
        Exception: If neither X_train or X_test is provided. 
    """
    equals = "=="*40
    header="\tCLASSIFICATION REPORT " + label
    dashes='--'*40
    
    # print(f"{dashes}\n{header}\n{dashes}")
    print(f"{equals}\n{header}\n{equals}")
    display(model)
    if (X_train is None) & (X_test is None):
        raise Exception("Must provide at least X_train & y_train or X_test and y_test")
    
    if (X_train is not None) & (y_train is not None):
        ## training data
        header ="[i] Training Data:"
        print(f"{dashes}\n{header}\n{dashes}")
        y_pred_train = model.predict(X_train)
        report_train = metrics.classification_report(y_train, y_pred_train)
        print(report_train)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
                                                      normalize=normalize, 
                                                      cmap=cmap,ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()

        plt.show()

    
        # print(dashes)

        
    if (X_test is not None) & (y_test is not None):
        ## training data
        header = f"[i] Test Data:"
        print(f"{dashes}\n{header}\n{dashes}")
        y_pred_test = model.predict(X_test)
        report_test = metrics.classification_report(y_test, y_pred_test)
        print(report_test)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
                                                      normalize=normalize, 
                                                      cmap=cmap, ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()
        plt.show()