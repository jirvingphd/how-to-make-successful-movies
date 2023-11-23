import matplotlib.pyplot as plt
import statsmodels.api as sm

def find_outliers_Z(data, verbose=True):
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    outliers = np.abs(stats.zscore(data))>3
    
    
    if verbose:
        n = len(outliers)
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum()/n*100:.2f}%) using Z-scores.")

    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers


def find_outliers_IQR(data, verbose=True):
    import pandas as pd
    import numpy as np
    q3 = np.quantile(data,.75)
    q1 = np.quantile(data,.25)

    IQR = q3 - q1
    upper_threshold = q3 + 1.5*IQR
    lower_threshold = q1 - 1.5*IQR
    
    outliers = (data<lower_threshold) | (data>upper_threshold)
    if verbose:
        n = len(outliers)
    
    
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum()/n*100:.2f}%) using IQR.")
        
    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers




def remove_outliers(df_,method='iqr', subset=None, verbose=2):
    """Returns a copy of the input df with outleirs removed from all
    columns using the selected method (either 'iqr' or 'z'/'zscore')
    
    Arguments:
        df_ (Frame): Dataframe to copy and remove outleirs from
        method (str): Method of outlier removal. Options are 'iqr' or 'z' (default is 'iqr')
        subset (list or None): List of column names to remove outliers from. If None, uses all numeric columns.
        verbose (bool, int): If verbose==1, print only overall summary. If verbose==2, print detailed summary"""
    import pandas as pd
    ## Make a cope of input dataframe  
    df = df_.copy()
    
    ## Set verbose_func for calls to outleir funcs
    if verbose==2:
        verbose_func = True
    else:
        verbose_func=False
        
    ## Set outlier removal function and name
    if method.lower()=='iqr':
        find_outlier_func = find_outliers_IQR
        method_name = "IQR rule"
    elif 'z' in method.lower():
        find_outlier_func = find_outliers_Z
        method_name = 'Z_score rule'
    else:
        raise Exception('[!] Method must be either "iqr" or "z".')
        
    ## Set list of cols to remove outliers from
    if subset is None:
        col_list = df.select_dtypes('number').columns
    elif isinstance(subset,str):
        col_list = [subset]
    elif isinstance(subset, list):
        col_list = subset
    else:
        raise Exception("[!] subset must be None, a single string, or a list of strings.")

    

    
    ## Empty dict for both types of outliers
    outliers = {}

    ## Use both functions to see the comparison for # of outliers
    for col in col_list:
        idx_outliers = find_outlier_func(df[col],verbose=verbose_func)
        outliers[col] = idx_outliers

    
    ## Getting final df of all outliers to get 1 final T/F index
    outliers_combined = pd.DataFrame(outliers).any(axis=1)
    
    if verbose:
        n = len(outliers_combined)
        print(f"\n[i] Overall, {outliers_combined.sum():,} rows out of {n:,}({outliers_combined.sum()/n*100:.2f}%) were removed as outliers using {method_name}.")
    
    
    # remove_outliers 
    df_clean = df[~outliers_combined].copy()
    return df_clean
      
    
    
# def evaluate_ols(result,X_train_df, y_train, show_summary=True):
#     """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
#     """
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm
#     try:
#         display(result.summary())
#     except:
#         pass
    
#     ## save residuals from result
#     y_pred = result.predict(X_train_df)
#     resid = y_train - y_pred
    
#     fig, axes = plt.subplots(ncols=2,figsize=(12,5))
    
#     ## Normality 
#     sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);
    
#     ## Homoscedasticity
#     ax = axes[1]
#     ax.scatter(y_pred, resid, edgecolor='white',lw=1)
#     ax.axhline(0,zorder=0)
#     ax.set(ylabel='Residuals',xlabel='Predicted Value');
#     plt.tight_layout()
    
    
    
# def plot_residuals(model,X_test_df, y_test,figsize=(12,5)):
#     """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
    
#     """
#     ## Make predictions and calculate residuals
#     y_pred = model.predict(X_test_df)
#     resid = y_test - y_pred
    
#     fig, axes = plt.subplots(ncols=2,figsize=figsize)
    
#     ## Normality 
#     sm.graphics.qqplot(resid, line='45',fit=True,ax=axes[0]);
    
#     ## Homoscedascity
#     ax = axes[1]
#     ax.scatter(y_pred, resid, edgecolor='white',lw=0.5)
#     ax.axhline(0,zorder=0)
#     ax.set(ylabel='Residuals',xlabel='Predicted Value');
#     plt.tight_layout()

from .evaluate import evaluate_ols, plot_residuals