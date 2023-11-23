## FROM MY "From linear to logistic regression" mini-lesson (interview)
from matplotlib import ticker
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def annotate_hbars(ax, ha='left',va='center',size=12,  xytext=(4,0),
                  textcoords='offset points'):
    for bar in ax.patches:

        ## get the value to annotate
        val = bar.get_width()

        if val<0:
            x=0
        else:
            x=val


        ## calculate center of bar
        bar_ax = bar.get_y() + bar.get_height()/2

        # ha and va stand for the horizontal and vertical alignment
        ax.annotate(f"{val:,.2f}", (x,bar_ax),ha=ha,va=va,size=size,
                    xytext=xytext, textcoords=textcoords)




from sklearn import metrics
import pandas as pd

# def evaluate_regression(model, X_train,y_train, X_test, y_test,as_frame=True):
#   """Evaluates a scikit learn regression model using r-squared and RMSE.
#   Returns the results a DataFrame if as_frame is True (Default).
#   """


#   ## Training Data
#   y_pred_train = model.predict(X_train)
#   r2_train = metrics.r2_score(y_train, y_pred_train)
#   rmse_train = metrics.mean_squared_error(y_train, y_pred_train,
#                                           squared=False)
#   mae_train = metrics.mean_absolute_error(y_train, y_pred_train)


#   ## Test Data
#   y_pred_test = model.predict(X_test)
#   r2_test = metrics.r2_score(y_test, y_pred_test)
#   rmse_test = metrics.mean_squared_error(y_test, y_pred_test,
#                                           squared=False)
#   mae_test = metrics.mean_absolute_error(y_test, y_pred_test)

#   if as_frame:
#       df_version =[['Split','R^2','MAE','RMSE']]
#       df_version.append(['Train',r2_train, mae_train, rmse_train])
#       df_version.append(['Test',r2_test, mae_test, rmse_test])
#       df_results = pd.DataFrame(df_version[1:], columns=df_version[0])
#       df_results = df_results.round(2)


#       # adapting hide_index for pd version
#       if pd.__version__ < "1.4.0":
#         display(df_results.style.hide_index().format(precision=2, thousands=','))
#       else:
#         display(df_results.style.hide(axis='index').format(precision=2, thousands=','))

#   else:
#       print(f"Training Data:\tR^2 = {r2_train:,.2f}\tRMSE = {rmse_train:,.2f}\tMAE = {mae_train:,.2f}")
#       print(f"Test Data:\tR^2 = {r2_test:,.2f}\tRMSE = {rmse_test:,.2f}\tMAE = {mae_test:,.2f}")



def get_coefficients(reg, name='Coefficients'):
    """Save a model's .coef_ and .intercept_ as a Pandas Series"""
    raise Exception("Deprecated - use get_coeffs_linreg instead")

    # coeffs = pd.Series(reg.coef_,
    #                    index= reg.feature_names_in_,
    #                    name=name)

    # if reg.intercept_ != 0.0:
    #     coeffs.loc['Intercept'] = reg.intercept_
    # return coeffs
    
def get_coeffs_linreg(lin_reg, feature_names = None, sort=True,ascending=True,
                     name='LinearRegression Coefficients'):
    if feature_names is None:
        feature_names = lin_reg.feature_names_in_
    ## Saving the coefficients
    coeffs = pd.Series(lin_reg.coef_, index= feature_names)
    coeffs['intercept'] = lin_reg.intercept_
    if sort==True:
        coeffs = coeffs.sort_values(ascending=ascending)
    return coeffs


def plot_coefficients(coeffs, figsize=(6,5), title='Regression Coefficients',
                      intercept=True, intercept_name='Intercept',
                      sort_values=True, ascending=True,
                      ):
    raise Exception("Deprecated: use plot_coeffs instead.")
#     ## Exclude intercept if intercept==False
#     if intercept==False:
#         if intercept_name in coeffs:
#             coeffs = coeffs.drop(intercept_name).copy()

#     ## Sort values
#     if sort_values:
#         ceoffs = coeffs.sort_values(ascending=ascending)

#     ## Plot
#     ax = ceoffs.plot(kind='barh',figsize=figsize)

#     ## Customize Viz
#     ax.axvline(0,color='k', lw=1)
#     ax.set(ylabel='Feature Name',xlabel='Coefficient',title=title)
#     return ax
def plot_coeffs(coeffs, top_n=None, figsize=(4,5), 
                intercept=False, intercept_name="intercept", 
                annotate=False, ha='left', va='center', size=12, 
                xytext=(4,0), textcoords='offset points'):
    """ Plots the top_n coefficients from a Series, with optional annotations.
    """
    # Drop intercept if intercept=False and 
    if (intercept == False) & (intercept_name in coeffs.index):
        coeffs = coeffs.drop(intercept_name)
    if top_n == None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients - Ranked by Magnitude"
    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)
        
        ## sort features and keep top_n and set title
        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        title = f"Top {top_n} Largest Coefficients"
    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Coefficient', 
            ylabel='Feature Names', 
            title=title)
    ax.axvline(0, color='k')
    if annotate == True:
        annotate_hbars(ax, ha=ha, va=va, size=size, xytext=xytext, textcoords=textcoords)
    return ax




def plot_residuals(model,X_test_df, y_test,figsize=(12,5)):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.

    """
    ## Make predictions and calculate residuals
    y_pred = model.predict(X_test_df)
    resid = y_test - y_pred

    fig, axes = plt.subplots(ncols=2,figsize=figsize)

    ## Normality
    sm.graphics.qqplot(resid, line='45',fit=True,ax=axes[0]);

    ## Homoscedascity
    ax = axes[1]
    ax.scatter(y_pred, resid, edgecolor='white',lw=0.5)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value');
    fig.tight_layout()
    plt.show()



def summarize_df(df_):
    """Source: Insights for Stakeholder Lesson - https://login.codingdojo.com/m/0/13079/91969
    Example Usage:
    >> df = pd.read_csv(filename)
    >> summary = summarize_df(df);
    """
    df = df_.copy()
    report = pd.DataFrame({
                        'dtype':df.dtypes,
                        '# null': df.isna().sum(),
                        'null (%)': df.isna().sum()/len(df)*100,
                        'nunique':df.nunique(),
                        "min":df.min(numeric_only=True),
                        'max':df.max(numeric_only=True)
             })
    report.index.name='Column'

    with pd.option_context("display.min_rows", len(df)):
        display(report.round(2))
    return report.reset_index()


def get_importances(model, feature_names=None,name='Feature Importance',
                   sort=False, ascending=True):

    ## checking for feature names
    if feature_names == None:
        feature_names = model.feature_names_in_

    ## Saving the feature importances
    importances = pd.Series(model.feature_importances_, index= feature_names,
                           name=name)

    # sort importances
    if sort == True:
        importances = importances.sort_values(ascending=ascending)

    return importances




def plot_importance(importances, top_n=None,  figsize=(8,6)):
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Importance',
            ylabel='Feature Names',
            title=title)
    ## return ax in case want to continue to update/modify figure
    return ax



def get_color_dict(importances, color_rest='#006ba4' , color_top='green',
                    top_n=7):
    ## color -coding top 5 bars
    highlight_feats = importances.sort_values(ascending=True).tail(top_n).index
    colors_dict = {col: color_top if col in highlight_feats else color_rest for col in importances.index}
    return colors_dict


def plot_importance_color(importances, top_n=None,  figsize=(8,6), 
                          color_dict=None, ax=None):
    """Formerly called `plot_importance_color_ax`

    Example Use:
    fig, axes = plt.subplots(ncols=2, figsize=(20,8))
    n = 20 # setting the # of features to use for both subplots
    
    plot_importance_color_ax(importances, top_n=n, ax=axes[0], color_dict= colors_top7)
    axes[0].set(title='R.F. Importances')

    plot_importance_color_ax(permutation_importances, top_n=n, ax=axes[1], color_dict=colors_top7)
    axes[1].set(title='Permutation Importances')
    fig.tight_layout()
    
    """
    
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"
    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"
    ## create plot with colors, if provided
    if color_dict is not None:
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        ax = plot_vals.plot(kind='barh', figsize=figsize, color=colors, ax=ax)
        
    else:
        ## create plot without colors, if not provided
        ax = plot_vals.plot(kind='barh', figsize=figsize, ax=ax)
        
    # set titles and axis labels
    ax.set(xlabel='Importance', 
           ylabel='Feature Names', 
           title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax


def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
                      name='LogReg Coefficients', class_index=0):
    if feature_names is None:
        feature_names = logreg.feature_names_in_ 
    
    ## Saving the coefficients
    coeffs = pd.Series(logreg.coef_[class_index],
                       index= feature_names, name=name)
    
    # use .loc to add the intercept to the series
    coeffs.loc['intercept'] = logreg.intercept_[class_index]
    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)  
    return coeffs



def plot_coeffs_color(coeffs, top_n=None,  figsize=(8,6), legend_loc='best',
                      threshold=None, color_lt='darkred', color_gt='forestgreen',
                      color_else='gray', label_thresh='Equally Likely',
                      label_gt='More Likely', label_lt='Less Likely',
                   plot_kws = {}):
    """Plots series of coefficients
        Args:
        ceoffs (pands Series): importance values to plot
        top_n (int): The # of features to display (Default=None).
                         If None, display all.
                        otherwise display top_n most important
                        
        figsize (tuple): figsize tuple for .plot
        color_dict (dict): dict with index values as keys with color to use as vals
                            Uses series.index.map(color_dict).
        plot_kws (dict): additional keyword args accepted by panda's .plot
        
         
         Returns:
        Axis: matplotlib axis
    """
    # sorting with asc=false for correct order of bars
    if top_n is None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients"
    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)
        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        ## sort features and keep top_n and set title
        title = f"Top {top_n} Largest Coefficients"
        ## plotting top N importances
    if threshold is not None:
        color_dict = get_colors_gt_lt(plot_vals, threshold=threshold,
                                      color_gt=color_gt,color_lt=color_lt,
                                      color_else=color_else)
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        plot_kws.update({'color':colors})
    
    
    ax = plot_vals.plot(kind='barh', figsize=figsize,**plot_kws)
    ax.set(xlabel='Coefficient',
            ylabel='Feature Names',
            title=title)
    
    if threshold is not None:
        ln1 = ax.axvline(threshold,ls=':',color='black')
        from matplotlib.patches import Patch
        box_lt = Patch(color=color_lt)
        box_gt = Patch(color=color_gt)
        handles = [ln1,box_gt,box_lt]
        labels = [label_thresh,label_gt,label_lt]
        ax.legend(handles,labels, loc=legend_loc)
    ## return ax in case want to continue to update/modify figure
    return ax


def get_colors_gt_lt(coeffs, threshold=1, color_lt ='darkred',
                     color_gt='forestgreen', color_else='gray'):
    """Creates a dictionary of features:colors based on if value is > or < threshold"""
    colors_dict = {}
    for i in coeffs.index:
        rounded_coeff = np.round( coeffs.loc[i],3)
        if rounded_coeff < threshold:
            color = color_lt
        elif rounded_coeff > threshold:
            color = color_gt
        else:
            color=color_else
        colors_dict[i] = color
    return colors_dict


