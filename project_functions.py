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

        
def savefig(fname,fig=None, ax=None,dpi=300,bbox_inches='tight',
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
	if ((fig==None) & (ax==None)) |((fig!=None) & (ax!=None)) :
		raise Exception("Must provide EITHER fig or AX")
		
	if fig is None:
		fig = ax.get_figure()

	fig.savefig(fname,dpi=dpi,bbox_inches=bbox_inches,facecolor=facecolor )
	if verbose:
		print(f'- Figure saved as {fname}')



def write_json(new_data, filename, return_data=False): 
	"""Appends the input json-compatible data to the json file.
	Adapted from: https://www.geeksforgeeks.org/append-to-json-file-using-python/

	Args:
		new_data (list or dict): json-compatible dictionary or list
		filename (str): json file to append data to
		
	Returns:
		return_data(bool): determines if combined data is returned (default =False) 
	"""
	import json
	with open(filename,'r+') as file:
		# First we load existing data into a dict.
		file_data = json.load(file)
		## Choose extend or append
		if (type(new_data) == list) & (type(file_data) == list):
			file_data.extend(new_data)
		else:
				file_data.append(new_data)
		# Sets file's current position at offset.
		file.seek(0)
		# convert back to json.
		json.dump(file_data, file)
		
	if return_data:
		return file_data




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
	
	
	

def evaluate_ols(result,X_train_df, y_train, figsize=(12,5), show_summary=True,return_fig=False):
	"""Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
	"""
	import matplotlib.pyplot as plt
	import statsmodels.api as sm
	
	if show_summary==True:
		try:
			display(result.summary())
		except:
			pass

	## save residuals from result
	y_pred = result.predict(X_train_df)
	resid = y_train - y_pred

	fig, axes = plt.subplots(ncols=2,figsize=figsize)

	## Normality 
	sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);

	## Homoscedasticity
	ax = axes[1]
	ax.scatter(y_pred, resid, edgecolor='white',lw=1)
	ax.axhline(0,zorder=0)
	ax.set(ylabel='Residuals',xlabel='Predicted Value');
	plt.tight_layout()
	if return_fig:
		return fig


def check_nulls_nunique(df,plot=True):
	"""Displays a df summary of the # & % of null values and unique values for each column.

	Args:
		df (DataFrame): Frame to check
		plot (bool, optional): Whether to plot missingo.matrixs. Defaults to True.
	"""
	import pandas as pd
	import matplotlib.pyplot as plt
	import missingno

	report = pd.DataFrame({"# null":df.isna().sum(),
				"% null":df.isna().sum()/len(df)*100,
				'# unique':df.nunique(),
						'% unique':df.nunique()/len(df)*100})
	display(report.round(2))
	if plot:
		missingno.matrix(df)
		plt.show()
		
		
		
def find_outliers_Z(data, verbose=True):
	"""Identifies outliers using Z-score > 3 rule.

	Args:
		data (pd.Series): data to check for outliers
		verbose (bool, optional): Print # of outliers in column. Defaults to True.

	Returns:
		Boolean Series: pd.Series with True/False for every value
	"""
	from scipy import stats
	import numpy as np
	outliers = np.abs(stats.zscore(data))>3

	if verbose:
		print(f"- {outliers.sum()} outliers found in {data.name} using Z-Scores.")
	return outliers
	
	
	
	
def find_outliers_IQR(data, verbose=True):
	"""Identifies outliers using IQR Rule. 
	Data that is more than 1.5*IQR less than Q1 or above Q3 is an outlier.
	

	Args:
		data (pd.Series): data to check for outliers
		verbose (bool, optional): Print # of outliers in column. Defaults to True.

	Returns:
		Boolean Series: pd.Series with True/False for every value
	"""
	import numpy as np
	q3 = np.quantile(data,.75)
	q1 = np.quantile(data,.25)

	IQR = q3 - q1
	upper_threshold = q3 + 1.5*IQR
	lower_threshold = q1 - 1.5*IQR

	outliers = (data<lower_threshold) | (data>upper_threshold)
	if verbose:
		print(f"- {outliers.sum()} outliers found in {data.name} using IQR.")
		
	return outliers
	
	
	
	
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