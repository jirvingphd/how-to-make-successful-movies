import os, json, requests


import pandas as pd

import scipy.stats as stats
import pandas as pd
import numpy as np


def Cohen_d(group1, group2, correction = False):
    """Compute Cohen's d
    d = (group1.mean()-group2.mean())/pool_variance.
    pooled_variance= (n1 * var1 + n2 * var2) / (n1 + n2)

    Args:
        group1 (Series or NumPy array): group 1 for calculating d
        group2 (Series or NumPy array): group 2 for calculating d
        correction (bool): Apply equation correction if N<50. Default is False. 
            - Url with small ncorrection equation: 
                - https://www.statisticshowto.datasciencecentral.com/cohens-d/ 
    Returns:
        d (float): calculated d value
         
    INTERPRETATION OF COHEN's D: 
    > Small effect = 0.2
    > Medium Effect = 0.5
    > Large Effect = 0.8
    
    """
    import scipy.stats as stats
    import scipy   
    import numpy as np
    N = len(group1)+len(group2)
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    ## Apply correction if needed
    if (N < 50) & (correction==True):
        d=d * ((N-3)/(N-2.25))*np.sqrt((N-2)/N)
    return d


#Your code here
def find_outliers_Z(data):
    """Use scipy to calculate absolute Z-scores 
    and return boolean series where True indicates it is an outlier.

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    ## Calculate z-scores
    zs = stats.zscore(data)
    
    ## Find z-scores >3 awayfrom mean
    idx_outs = np.abs(zs)>3
    
    ## If input was a series, make idx_outs index match
    if isinstance(data,pd.Series):
        return pd.Series(idx_outs,index=data.index)
    else:
        return pd.Series(idx_outs)
    
    
    
def find_outliers_IQR(data):
    """Use Tukey's Method of outlier removal AKA InterQuartile-Range Rule
    and return boolean series where True indicates it is an outlier.
    - Calculates the range between the 75% and 25% quartiles
    - Outliers fall outside upper and lower limits, using a treshold of  1.5*IQR the 75% and 25% quartiles.

    IQR Range Calculation:    
        res = df.describe()
        IQR = res['75%'] -  res['25%']
        lower_limit = res['25%'] - 1.5*IQR
        upper_limit = res['75%'] + 1.5*IQR

    Args:
        data (Series,or ndarray): data to test for outliers.

    Returns:
        [boolean Series]: A True/False for each row use to slice outliers.
        
    EXAMPLE USE: 
    >> idx_outs = find_outliers_df(df['AdjustedCompensation'])
    >> good_data = df[~idx_outs].copy()
    
    """
    df_b=data
    res= df_b.describe()

    IQR = res['75%'] -  res['25%']
    lower_limit = res['25%'] - 1.5*IQR
    upper_limit = res['75%'] + 1.5*IQR

    idx_outs = (df_b>upper_limit) | (df_b<lower_limit)

    return idx_outs
    

def simulate_tukeys_results(groups, result):
    """Construct a tukeys-results-like table for a dictionary containing 2-groups"""
    results_to_annotate =  {}
    for i, group in enumerate(groups.keys()):
        results_to_annotate[f"group{i+1}"] = group
        
    results_to_annotate['p-adj'] = result.pvalue
    results_to_annotate['reject'] = result.pvalue <.05
    results_to_annotate_df = pd.DataFrame(results_to_annotate, index=[0])
    return results_to_annotate_df



def prep_data_for_tukeys(data, group_col = 'group', values_col = 'data'):
    """Accepts a dictionary with group names as the keys 
    and pandas series as the values. 
    
    Returns a dataframe ready for tukeys test:
    - with a 'data' column and a 'group' column for sms.stats.multicomp.pairwise_tukeyhsd 
    
    Example Use:
    df_tukey = prep_data_for_tukeys(grp_data)
    tukey = sms.stats.multicomp.pairwise_tukeyhsd(df_tukey['data'], df_tukey['group'])
    tukey.summary()
    """
    
    df_tukey = pd.DataFrame(columns=[values_col, group_col])
    for k,v in  data.items():
        grp_df = v.rename(values_col).to_frame() 
        grp_df[group_col] = k
        df_tukey=pd.concat([df_tukey, grp_df],axis=0)

	## New lines added to ensure compatibility with tukey's test
    df_tukey[group_col] = df_tukey[group_col].astype('str')
    df_tukey[values_col] = df_tukey[values_col].astype('float')
    return df_tukey


def check_assumptions_normality(groups_dict, alpha=.05, as_markdown=False):
    """
    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.
    """
    import pandas as pd
    from scipy import stats
    ## Running normal test on each group and confirming there are >20 in each group
    results = []
    for group_name, group_data in groups_dict.items():
        try:
            stat, p = stats.shapiro(group_data)
            test_name = 'Shapiro-Wilk (Normality)'
        except:
            print(f'[!] Error with {group_name}')
            p = np.nan
            
        ## save the p val, test statistic, and the size of the group
        results.append({#'stat test':test_name, 
                        'group':group_name, 
                        'n': len(group_data),
                         'stat':f"{stat:.5f}",
                        'p':p,#f"{p:.10f}",
                        'p (.4)':f"{p:.4f}",
                        'sig?': p<alpha})

    results_df = pd.DataFrame(results).sort_values('group', ascending=True)

    if as_markdown == True:
        results_df = results_df.to_markdown(index=False)   
        print(results_df)
    else:
        return results_df#.set_index('stat test')
        



    
    # if as_markdown == True:

    #     results_df = normal_results_to_markdown(results_df, sort_by='group')
    #     print(results_df)
    # else:
    #     results_df = pd.DataFrame(results).set_index('stat test')#,"group"])

    #     return results_df


# # Prepare Normality Resuls as markdown
# def normal_results_to_markdown(normal_results, sort_by='group'):
#     normal_results = normal_results.reset_index()
#     normal_results = normal_results.drop(columns=['stat test'])
#     normal_results.sort_values(by=sort_by)
    
#     return normal_results.to_markdown(index=False)




def remove_and_display_outliers(groups, as_markdown=True):
    groups_cleaned = {}

    outlier_results = []

    for group_name, data in groups.items():
    
        outliers = np.abs(stats.zscore(data)) > 3

        

        # print(f"There were {} ({outliers.sum()/len(outliers)*100:.2f}%) outliers in the {sector} group.")
    
        group_data = data.loc[~outliers]
        groups_cleaned[group_name] = group_data

        outlier_results.append({'group':group_name,
                               'n (original)': len(data),
                               '# outliers': outliers.sum(),
                               "% outliers":f"{outliers.sum()/len(outliers)*100: .2f}%",
                                "n (final)": len(group_data)
                               })
        
    # Formatting results
    results_df = pd.DataFrame(outlier_results)
    results_df = results_df.sort_values('group', ascending=True)
        
    if as_markdown == True:
        results_df = results_df.to_markdown(index=False)   
        print(results_df)
    else:
        display(results_df)
    return groups_cleaned#, results_df





def add_sig_legend(ax, frame=False, bbox_to_anchor=(1,1), title='Significance Levels', marker='',
                  color=None, alignment='center'):
    from matplotlib.lines import Line2D
    sig_dict= {"***":"p < .001",
               "**":"p < .01",
        '*':"p < .05"}
    
    legend_elements = []
    for asts, pval in sig_dict.items():
        label =  f'{asts:<5}  {pval:>10}'
        legend_elements.append( Line2D([0], [0], marker=marker, color=color, markerfacecolor='k',linewidth=0, markersize=0, label=label))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, #loc='upper right',
              bbox_to_anchor = bbox_to_anchor,
              frameon=frame,title=title, alignment=alignment)



def annotate_tukey_significance(ax, tukey_results, delta_pad_line = 0.06, 
                                delta_pad_text=0.02, delta_err = 1.05,
                                linewidth=.75, fontsize=8, legend=True):
    """
    Annotate a barplot with Tukey's HSD test significance, ensuring annotations do not overlap.

    Parameters:
    - ax: The matplotlib axis object containing the barplot.
    - tukey_results: DataFrame with columns ['group1', 'group2', 'p-adj', 'reject'].
    - group_order: List of groups in the same order as they appear in the barplot.
    - delta_pad_line: Proportional padding above each annotation line.
    - delta_pad_text: Proportional padding above the text relative to its line.
    - delta_err: Proportional padding for first annotation relative to error bar.
    - linewidth: Width of the annotation lines.
    - fontsize: Size of the annotation text.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker

    # Get results as a dataframe
    if not isinstance(tukey_results, pd.DataFrame):
        raw_tukey_data = tukey_results._results_table.data
        tukey_results = pd.DataFrame(raw_tukey_data[1:],columns=raw_tukey_data[0])


    # Determine the height of the highest errorbar
    highest_bar = max([p.get_height() for p in ax.patches])
    highest_errorbar = highest_bar * delta_err#1.1  # Assuming errorbar might add 10% height, adjust this based on your error calculation
    
        
    # Determine the initial y-axis height to start the annotations
    initial_y_max = highest_errorbar#ax.get_ylim()[1]
    y_max = initial_y_max * delta_err#1.1
    ax.set_ylim(top=y_max)
    
    # Fixed distance above the last annotation for the next one
    delta = initial_y_max * delta_pad_line#0.06  # Adjust this value based on your plot's scale
    
    # Variable to keep track of the last annotation's height
    last_annotation_height = y_max

    # Extract group order from x-axis labels
    group_order = [tick.get_text() for tick in ax.get_xticklabels()]
    
    # Iterate through Tukey's test results
    for _, row in tukey_results.iterrows():
        if row['reject']:  # If the result is significant
            # Find the positions of the groups on the x-axis
            group1_pos = group_order.index(row['group1'])
            group2_pos = group_order.index(row['group2'])
            x_mid = (group1_pos + group2_pos) / 2  # Midpoint for the annotation

            # Determine significance level for annotation
            if row['p-adj'] < 0.001:
                sig = '***'
            elif row['p-adj'] < 0.01:
                sig = '**'
            elif row['p-adj'] < 0.05:
                sig = '*'
            else:
                sig = ''  # No asterisks for non-significant differences
            
            # Calculate the height for the annotation, ensuring it's above the last one
            y = last_annotation_height + delta  # Increment from last height
            
            # Draw the line and the annotation
            ax.plot([group1_pos, group2_pos], [y, y], lw=linewidth, color='black')
            ax.text(x_mid, y + (delta * delta_pad_text), sig, ha='center', va='bottom', fontweight='semibold',fontsize=fontsize)  # Slightly above the line
            
            # Update the last_annotation_height to the current y position
            # last_annotation_height = y
            last_annotation_height += delta + (delta * delta_pad_text)


    # Adjust the plot's ylim to accommodate the last annotation
    ax.set_ylim(top=last_annotation_height + delta)

    if legend:
        add_sig_legend(ax)
        


def plot_simultaneous_comparison( tukeys_results, compare_group = None,compare_color='blue', figsize=(10,10), ):
    """Plot a tukey's result plot_simultaneous with option to use "compare_group" to selectively color the group blue.
    Any significantly different groups will appear in red.
    If compare_group is not none, the name of the group is also annotated in blue.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    tukeys_results.plot_simultaneous(ylabel="Genre", xlabel="Group Distributions",ax=ax, comparison_name=compare_group);
    if compare_group is not None:
        
        ax.set_title(f"Tukey Pairwise Multiple Comparison Results ({compare_group} vs All)");
        # Highlighting the specified group
        for label in ax.get_yticklabels():
            if label.get_text() == compare_group:
                label.set_color(compare_color)  # Change color to red for the compare_group

    return fig, ax



def annotate_bars(ax,fmt='.2f',size=15,xytext=(0,8),ha='center', va='center',
                  convert_millions=False, despine=False, spines = ['right','top'],
				  use_errorbars=False):
	"""Annotates bar values above each bar
	Adapted from: https://www.geeksforgeeks.org/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/

	Args:
		ax (matplotlib axes): ax containing patches to annotate (bars)
		fmt (str): string format code for number format. Default='.2f'
		size (int): text size in pts. Default=15
		xytest (tuple): Padding for annotations (in offset points). Default=(0,8)
		ha (str): horizontal alignment. Default is 'center'
		va (str): vertical alignment. Default is 'center'
		convert_millions(bool): Determines if values are calculated as Millions
	"""
	import numpy as np
	if use_errorbars==True:
		err_height= get_line_data(ax,just_error_heights=True)
		patches_errors = list(zip(ax.patches,err_height))
		
	else:
		patches_errors = [(patch,None) for patch in ax.patches]
	# Iterrating over the bars one-by-one
	
	for bar, err in patches_errors:#ax.patches:
		# Using Matplotlib's annotate function and
		# passing the coordinates where the annotation shall be done
		# x-coordinate: bar.get_x() + bar.get_width() / 2
		# y-coordinate: bar.get_height()
		# free space to be left to make graph pleasing: (0, 8)
		# ha and va stand for the horizontal and vertical alignment
		
		if use_errorbars==False:
			if convert_millions==False:
				height = bar.get_height()
				value = format(height,fmt )
				
			else:
				height = bar.get_height()
				value = format(height/1_000_000,fmt)+ "M"
		else:
			if convert_millions==False:
				height = err
				value = format(err,fmt )
			else:
				height = err#/1_000_000
				value = format(height/1_000_000,fmt)+ "M"						

		
		ax.annotate(value,
						(bar.get_x() + bar.get_width() / 2,
						height),#bar.get_height()), 
					ha=ha, va=va,
						size=size, xytext=xytext,
						textcoords='offset points')
					

		
	if despine:
		## removing top and right border
		for side in spines:
			ax.spines[side].set_visible(False)
	return ax


def get_line_data(ax,just_error_heights=False):
	"""Adapted From Source: https://stackoverflow.com/a/46271417"""
	import numpy as np
	x_list = []
	lower_list = []
	upper_list = []
	for line in ax.lines:
		x_list.append(line.get_xdata()[0])
		lower_list.append(line.get_ydata()[0])
		upper_list.append(line.get_ydata()[1])
		
	y = 0.5 * (np.asarray(lower_list) + np.asarray(upper_list))
	y_error = np.asarray(upper_list) - y
	x = np.asarray(x_list)

	if just_error_heights==False:
		return x, y, y_error
	else: 
		return upper_list

        
def savefig(fname,fig=None, ax=None,dpi=300, bbox_inches='tight',
            facecolor='auto' ,verbose=True):
    """Saves matplotlib fig using either fig or ax for plot to save.
    
    Args:
        fname (str): image filename (ending with extension  (e.g. .png))
        fig (matplotlib Figure, optional): figure object to save. Defaults to None.
        ax (mayplotlib Axes), optional): ax of of figure object to save. Defaults to None.
        dpi (int, optional): pixel density. Defaults to 300.
        bbox_inches (str, optional): corrects cutoff labels. Defaults to 'tight'.
        facecolor (str, optional): control figure facecolor. Defaults to 'auto'.
        verbose (bool, optional): Print filepath of saved iamge. Defaults to True.
    
    Raises:
        Exception: If BOTH fig and ax are passed OR neither fig or ax are passed.
    """
    import os
    dir = os.path.dirname(fname)
    os.makedirs(dir, exist_ok=True)
    
    if ((fig==None) & (ax==None)) |((fig!=None) & (ax!=None)) :
        raise Exception("Must provide EITHER fig or AX")
        
    if fig is None:
        fig = ax.get_figure()
    
    fig.savefig(fname,dpi=dpi,bbox_inches=bbox_inches,facecolor=facecolor )
    if verbose:
        print(f'- Figure saved as {fname}')



# def write_json(new_data, filename, return_data=False): 
# 	"""Appends the input json-compatible data to the json file.
# 	Adapted from: https://www.geeksforgeeks.org/append-to-json-file-using-python/

# 	Args:
# 		new_data (list or dict): json-compatible dictionary or list
# 		filename (str): json file to append data to
		
# 	Returns:
# 		return_data(bool): determines if combined data is returned (default =False) 
# 	"""
# 	import json
# 	with open(filename,'r+') as file:
# 		# First we load existing data into a dict.
# 		file_data = json.load(file)
# 		## Choose extend or append
# 		if (type(new_data) == list) & (type(file_data) == list):
# 			file_data.extend(new_data)
# 		else:
# 				file_data.append(new_data)
# 		# Sets file's current position at offset.
# 		file.seek(0)
# 		# convert back to json.
# 		json.dump(file_data, file)
		
# 	if return_data:
# 		return file_data



def write_json(new_data, filename, overwrite=True, skip=False): 
    """Appends a list of records (new_data) to a json file (filename). 
    Adapted from: https://www.geeksforgeeks.org/append-to-json-file-using-python/"""  
    
    if (overwrite==True) & (skip==True):
        raise Exception("Only 1 of overwrite or skip may be set to True")

    # find if file exists
    exists = os.path.exists(filename)

    if exists==False:
        mode = 'w'
    else:
        if overwrite==True:
            # exists=False
            mode = 'w'
            
        elif skip == True:
            return 
            
        else:
            mode = 'r+'

    
    with open(filename,mode) as file:

        if mode == 'r+':
            # First we load existing data into a dict.
            file_data = json.load(file)
            
            ## Choose extend or append
            if (type(new_data) == list) & (type(file_data) == list):
                file_data.extend(new_data)
            else:
                 file_data.append(new_data)
            # Sets file's current position at offset.
            file.seek(0)
        else:
            file_data = new_data
            
        # convert back to json.
        json.dump(file_data, file)
        return


def millions(x,pos,prefix='$', suffix=None,float_fmt=","):
    """function for use wth matplotlib FuncFormatter -  formats money in millions"""
    x = x*1e-6
    if suffix is None:
        suffix="M"
    string = "{prefix}{x:"+float_fmt+"}{suffix}"
    return string.format(prefix=prefix,x=x, suffix=suffix)

# Create the formatter
# price_fmt_mill =FuncFormatter(millions)
# ## Set the axis' major formatter
# ax.yaxis.set_major_formatter(price_fmt_mill)

def billions(x,pos,prefix='$', suffix=None, float_fmt=','):
	"""function for use wth matplotlib FuncFormatter -  formats money in billions"""
	x = x*1e-9
	if suffix is None:
		suffix='B'
	string = "{prefix}{x:"+float_fmt+"}{suffix}"
	return string.format(prefix=prefix,x=x, suffix=suffix)
	# return f"{prefix}{x*1e-9:,}{suffix}"


def get_funcformatter(kind='m',prefix='$',suffix=None, float_fmt=','):
	"""Returns a matplotlib FuncFormatter for formatting currecny in millions or billions

	Args:
		kind (str): which order of magnitude to use. Default is 'm'. 
					m=Millions, b=Billions
					
	EXAMPLE:
	>> ax = sns.barplot(data=movies, x='certification',y='revenue')
	>> price_fmt_m = get_funcformatter(kind='m')
	>> ax.yaxis.set_major_formatter(price_fmt_m)
	"""
	from matplotlib.ticker import FuncFormatter

	if kind.lower()=='m':
		func = lambda x,pos: millions(x,pos,prefix=prefix,suffix=suffix, float_fmt=float_fmt)
	elif kind.lower()=='b':
		func = lambda x, pos: billions(x,pos,prefix=prefix,suffix=suffix, float_fmt=float_fmt)
	return FuncFormatter(func)
	
	
	

# def evaluate_ols(result,X_train_df, y_train, figsize=(12,5), show_summary=True,return_fig=False):
# 	"""Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
# 	"""
# 	import matplotlib.pyplot as plt
# 	import statsmodels.api as sm
	
# 	if show_summary==True:
# 		try:
# 			display(result.summary())
# 		except:
# 			pass

# 	## save residuals from result
# 	y_pred = result.predict(X_train_df)
# 	resid = y_train - y_pred

# 	fig, axes = plt.subplots(ncols=2,figsize=figsize)

# 	## Normality 
# 	sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);

# 	## Homoscedasticity
# 	ax = axes[1]
# 	ax.scatter(y_pred, resid, edgecolor='white',lw=1)
# 	ax.axhline(0,zorder=0)
# 	ax.set(ylabel='Residuals',xlabel='Predicted Value');
# 	plt.tight_layout()
# 	if return_fig:
# 		return fig


# def check_nulls_nunique(df,plot=True):
# 	"""Displays a df summary of the # & % of null values and unique values for each column.

# 	Args:
# 		df (DataFrame): Frame to check
# 		plot (bool, optional): Whether to plot missingo.matrixs. Defaults to True.
# 	"""
# 	import pandas as pd
# 	import matplotlib.pyplot as plt
# 	import missingno

# 	report = pd.DataFrame({"# null":df.isna().sum(),
# 				"% null":df.isna().sum()/len(df)*100,
# 				'# unique':df.nunique(),
# 						'% unique':df.nunique()/len(df)*100})
# 	display(report.round(2))
# 	if plot:
# 		missingno.matrix(df)
# 		plt.show()
		
		
		
# def find_outliers_Z(data, verbose=True):
# 	"""Identifies outliers using Z-score > 3 rule.

# 	Args:
# 		data (pd.Series): data to check for outliers
# 		verbose (bool, optional): Print # of outliers in column. Defaults to True.

# 	Returns:
# 		Boolean Series: pd.Series with True/False for every value
# 	"""
# 	from scipy import stats
# 	import numpy as np
# 	outliers = np.abs(stats.zscore(data))>3

# 	if verbose:
# 		print(f"- {outliers.sum()} outliers found in {data.name} using Z-Scores.")
# 	return outliers
	
	
	
	
# def find_outliers_IQR(data, verbose=True):
# 	"""Identifies outliers using IQR Rule. 
# 	Data that is more than 1.5*IQR less than Q1 or above Q3 is an outlier.
	

# 	Args:
# 		data (pd.Series): data to check for outliers
# 		verbose (bool, optional): Print # of outliers in column. Defaults to True.

# 	Returns:
# 		Boolean Series: pd.Series with True/False for every value
# 	"""
# 	import numpy as np
# 	q3 = np.quantile(data,.75)
# 	q1 = np.quantile(data,.25)

# 	IQR = q3 - q1
# 	upper_threshold = q3 + 1.5*IQR
# 	lower_threshold = q1 - 1.5*IQR

# 	outliers = (data<lower_threshold) | (data>upper_threshold)
# 	if verbose:
# 		print(f"- {outliers.sum()} outliers found in {data.name} using IQR.")
		
# 	return outliers
	
	
	
	
def read_and_fix_json(JSON_FILE):
    """Attempts to read in json file of records and fixes the final character
    to end with a ] if it errors.
    
    Args:
        JSON_FILE (str): filepath of JSON file
        
    Returns:
        DataFrame: the corrected data from the bad json file
    """
    try: 
        previous_df =  pd.read_json(JSON_FILE)
    
    ## If read_json throws an error
    except:
        
        ## manually open the json file
        with open(JSON_FILE,'r+') as f:
            ## Read in the file as a STRING
            bad_json = f.read()
            
            ## if the final character doesn't match first, select the right bracket
            first_char = bad_json[0]
            final_brackets = {'[':']', 
                           "{":"}"}
            ## Select expected final brakcet
            final_char = final_brackets[first_char]
            
            ## if the last character in file doen't match the first char, add it
            if bad_json[-1] != final_char:
                good_json = bad_json[:-1]
                good_json+=final_char
            else:
                raise Exception('ERROR is not due to mismatched final bracket.')
            
            ## Rewind to start of file and write new good_json to disk
            f.seek(0)
            f.write(good_json)
           
        ## Load the json file again now that its fixed
        previous_df =  pd.read_json(JSON_FILE)
        
    return previous_df


def request_reviews(movie_id,login,page=1):
    import requests
    # source: https://developer.themoviedb.org/reference/movie-reviews
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?language=en-US&page={page}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {login['api-token']}"
    }
    response = requests.get(url, headers=headers)
    try:
        return response.json()
    except:
        print("[!] Error returning response.json()")
        return response 
    # print(response.text)

def get_reviews(movie_id,login,start_page=1):
    
    reviews_json = request_reviews(movie_id, page=start_page)
    try:
        n_pages = reviews_json['total_pages']
    except:
        n_pages=1
        
    all_responses = reviews_json['results']
    
    for page in range(2, n_pages+1):
        reviews_json = request_reviews(movie_id, page=page)
        all_responses.extend(reviews_json['results'])

    ## Add movie id to all results
    final_results = []
    for review in all_responses:
        
        review_results = {
                    "movie_id": movie_id, #review["movie_id"],
                    "review_id": review["id"],
                    "author_rating": review["author_details"]["rating"],
                    "review_text": review["content"],
                    "created_at": review['created_at'],
                    # 'updated_at':review['updated_at']
        }
    
        final_results.append(review_results)
        
    return final_results