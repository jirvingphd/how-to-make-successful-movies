import matplotlib.pyplot as plt
import seaborn as sns

####### PREVIOUS (slightly updated to only reuturn fig)

def summarize_df(df_):
    """Source: Insights for Stakeholder Lesson - https://login.codingdojo.com/m/0/13079/91969 
    Example Usage:
    >> df = pd.read_csv(filename)
    >> summary = summarize_df(df)
    >> summary"""
    import pandas as pd
    df = df_.copy()
    report = pd.DataFrame({
                        'dtype':df.dtypes,
                        '# null': df.isna().sum(),
                        'null (%)': df.isna().sum()/len(df)*100,
                        'nunique':df.nunique(),
                        "min":df.min(),
                        'max':df.max()
             })
    report.index.name='Column'
    return report.reset_index()
  
  
  
def explore_numeric(df, x, figsize=(6,5) ):
  """Plots a Seaborn histplot on the top subplot and a horizontal boxplot on he bottom.
	Additionally, prints information on: 
	- the # and % of null values
	- number of unique values
	- the most frequent value and how often frequent it is (%)
	- A warning message if the feature is quasi-constant or constant feature
							(if more than 99% of feature is a single value)

	Args:
		df (Frame): DataFrame that contains column x
		x (str): a column name 
		fillna (bool, optional): if True, fillna with the placeholder. Defaults to True.
		placeholder (str, optional): Value used to fillna if fillna is True. Defaults to 'MISSING'.
		figsize (tuple, optional): Figure size (width, height). Defaults to (6,5).
		order (list, optional): List of categories to include in graph, in the specified order. Defaults to None. 
								Note: any category not in the order list will not be shown on the graph.
								    If a category is included in the order list that isn't in the data, 
									it will be added as an empty bar categories can be removed from the visuals 

	Returns:
		fig: Matplotlib Figure
		ax: Matplotlib Axes
	
  Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()

  # Ensure plot is shown before message
  plt.show()
  ## Print message with info on the count and % of null values
  null_count = df[x].isna().sum()
  null_perc = null_count/len(df)* 100
  print(f"- NaN's Found: {null_count} ({round(null_perc,2)}%)")
  return fig, axes




def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
	"""Plots a seaborn countplot of for x column and prints information on:
	- the # and % of null values
	- number of unique values
	- the most frequent category  and how much of the feature is this category (%)
	- A warning message if the feature is quasi-constant or constant feature
							(if more than 99% of feature is a single value)

	Args:
		df (Frame): DataFrame that contains column x
		x (str): a column name 
		fillna (bool, optional): if True, fillna with the placeholder. Defaults to True.
		placeholder (str, optional): Value used to fillna if fillna is True. Defaults to 'MISSING'.
		figsize (tuple, optional): Figure size (width, height). Defaults to (6,4).
		order (list, optional): List of categories to include in graph, in the specified order. Defaults to None. 
								Note: any category not in the order list will not be shown on the graph.
								    If a category is included in the order list that isn't in the data, 
									it will be added as an empty bar categories can be removed from the visuals 

	Returns:
		fig: Matplotlib Figure
		ax: Matplotlib Axes
	"""
	# Make a copy of the dataframe and fillna 
	temp_df = df.copy()


	## Save null value counts and percent for printing 
	null_count = temp_df[x].isna().sum()
	null_perc = null_count/len(temp_df)* 100


	# fillna with placeholder
	if fillna == True:
		temp_df[x] = temp_df[x].fillna(placeholder)


	# Create figure with desired figsize
	fig, ax = plt.subplots(figsize=figsize)

	## Plotting a count plot 
	sns.countplot(data=temp_df, x=x, ax=ax, order=order)

	# Rotate Tick Labels for long names
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

	# Add. atitle with the feature name included
	ax.set_title(f"Column: {x}", fontweight='bold')

	# Fix layout and show plot (before print statements)
	fig.tight_layout()
	plt.show()


	# Print null value info
	print(f"- NaN's Found: {null_count} ({round(null_perc,2)}%)")
	# Print cardinality info
	nunique = temp_df[x].nunique()
	print(f"- Unique Values: {nunique}")


	# Get the most most common value, its count as # and as %
	most_common_val_count = temp_df[x].value_counts(dropna=False).head(1)
	most_common_val = most_common_val_count.index[0]
	freq = most_common_val_count.values[0]

	perc_most_common = freq / len(temp_df) * 100
	print(f"- Most common value: '{most_common_val}' occurs {freq} times ({round(perc_most_common,2)}%)")

	# print message if quasi-constant or constant (most common val more than 98% of data)
	if perc_most_common > 98:
		print(f"\n- [!] Warning: '{x}' is a constant or quasi-constant feature and should be dropped.")

	return fig, ax



def plot_categorical_vs_target(df, x, y,
                                   target_type='reg',
                                   figsize=(6,4),
                                   fillna = True, placeholder = 'MISSING',
                                   order = None,
                                   ):
  """Updated Version of the function which accepts either numeric or categorical targets.
  Adapted from Source: https://login.codingdojo.com/m/606/13765/117606
  Plots a combination seaborn barplot (without error bars) and a stripplot.

	Args:
		df (Frame): DataFrame containing data to plot.
		x (str): Column to use as the x-axis (categories)
		y (str, optional): Target column to plot on the y-axis. Defaults to 'SalePrice'.		
		fillna (bool, optional): if True, fillna with the placeholder. Defaults to True.
		placeholder (str, optional): Value used to fillna if fillna is True. Defaults to 'MISSING'.
		figsize (tuple, optional): Figure size (width, height). Defaults to (6,4).
		order (list, optional): List of categories to include in graph, in the specified order. Defaults to None. 
								Note: any category not in the order list will not be shown on the graph.
								    If a category is included in the order list that isn't in the data, 
									it will be added as an empty bar categories can be removed from the visuals 

	Returns:
		fig: Matplotlib Figure
		ax: Matplotlib Axes
	"""
 
  # Make a copy of the dataframe and fillna
  temp_df = df.copy()
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)

  # or drop nulls prevent unwanted 'nan' group in stripplot
  else:
    temp_df = temp_df.dropna(subset=[x])


  # Create the figure and subplots
  fig, ax = plt.subplots(figsize=figsize)

  ## If a regression target:
  if 'reg' in target_type:

      # Barplot
    sns.barplot(data=temp_df, x=x, y=y, ax=ax, order=order, alpha=0.6,
                linewidth=1, edgecolor='black', errorbar=None)

    # Boxplot
    sns.stripplot(data=temp_df, x=x, y=y, hue=x, ax=ax,
                  order=order, hue_order=order, legend=False,
                  edgecolor='white', linewidth=0.5,
                  size=3,zorder=0)
    # Rotate xlabels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


  # If a classification target:
  elif 'class' in target_type:
   sns.histplot(data=df, hue=y, x=x,hue_order=order,
                  stat='percent', multiple='fill',ax=ax)
  else:
    raise Exception("target_type must be one either 'class' or 'reg'")

  # Final Plot customization
  # Add a title
  ax.set_title(f"{x} vs. {y}", fontweight='semibold')
  fig.tight_layout()
  return fig, ax



def plot_numeric_vs_target(df, x, y, figsize=(6,4),
                           target_type='reg', errorbar='ci',
                           estimator='mean', order=None,
                           **kwargs): # kwargs for sns.regplot
  """UPDATED FUNCTION WITH OPTION FOR WHICH TYPE OF TARGET
  Source: https://login.codingdojo.com/m/606/13765/117605
  Plots a seaborn regplot, with an optional formula annotation.
	Also calcualtes correlation and displays Pearson's r in the title.

	Args:
		df (Frame): DataFrame with data.
		x (str): Numeric column name.
		y (str, optional): Numeric target column name. Defaults to 'SalePrice'.
		figsize (tuple, optional): Figure size. Defaults to (6,4).
		annotate (bool, optional): Whether to annotate regplot equation. Defaults to False. 
		
	Returns:
		fig: Matplotlib Figure
		ax: Matplotlib Axes
  """

  nulls = df[[x,y]].isna().sum()
  if nulls.sum()>0:
    print(f"- Excluding {nulls.sum()} NaN's")
    # print(nulls)
    temp_df = df.dropna(subset=[x,y,])
  else:
    temp_df = df

  if 'reg' in target_type:
    fig, axes = plt.subplots(figsize=figsize)
    # Calculate the correlation
    corr = df[[x,y]].corr().round(2)
    r = corr.loc[x,y]
    # Plot the data
    scatter_kws={'ec':'white','lw':1,'alpha':0.8}
    sns.regplot(data=temp_df, x=x, y=y, ax=axes, scatter_kws=scatter_kws, **kwargs) # Included the new argument within the sns.regplot function
    ## Add the title with the correlation
    axes.set_title(f"{x} vs. {y} (r = {r})", fontweight='bold')


  elif 'class' in target_type:
    fig, axes = plt.subplots(figsize=figsize, ncols=2)

    # Left Subplot (barplot)
    sns.barplot(data=temp_df, x=y, y=x,  order=order, estimator=estimator,
                errorbar=errorbar, ax=axes[0],)

    ## Right subplot (boxplot+stripplot)
    # Stripplot
    sns.stripplot(data=temp_df, x=y, y=x, hue=y, ax=axes[1],
                order=order, hue_order=order, legend=False,
                edgecolor='white', linewidth=0.5,
                size=3,zorder=0)

    # Boxplot
    transparent = {'alpha':.6} #Props for boxplot
    sns.boxplot(data=temp_df, x=y, y=x,
                boxprops=transparent, whiskerprops=transparent,
                width=.25, showfliers=False,
                saturation=0.5,
                ax=axes[1])
    # Add title
    fig.suptitle(f"{x} vs. {y}")

  # Make sure the plot is shown before the print statement
  fig.tight_layout()
  fig.show()
  return fig, axes



######### NEW
def plot_correlation(df, cmap='coolwarm', cols=None):
    if cols == None:
        cols = df.columns
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap=cmap, ax=ax, annot=True, center=0)
    ax.set_title("Correlation Matrix")
    return fig


from ._eda_functions_plotly import *



def annotate_regplot_equation(ax):
  """Adapted from Source: https://www.statology.org/seaborn-regplot-equation/
  Example Use:
  >> fig, ax = plot_numeric_vs_target(df, x="Living Area Sqft")
  >> annotate_regplot_equation(ax)
  """
  import scipy
  #calculate slope and intercept of regression equation
  slope, intercept, r, p, sterr = scipy.stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                        y=ax.get_lines()[0].get_ydata())
  eqn = f'y = {slope:,.2f} * X + {intercept:,.2f}'
  ax.legend(handles=[ax.get_lines()[0]], labels=[eqn])