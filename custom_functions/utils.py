
def reference_set_seed_keras(markdown=True):
    
    ref = """
    ```python
    # From source: https://keras.io/examples/keras_recipes/reproducibility_recipes/
    import tensorflow as tf
    import numpy as np

    # Then Set Random Seeds
    tf.keras.utils.set_random_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    # Then run the Enable Deterministic Operations Function
    tf.config.experimental.enable_op_determinism()
    ```
    """
    if markdown:
        from IPython.display import display, Markdown
        display(Markdown(ref))
    else:
        print(ref)
     
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import pandas as pd


## Dataset prep
def preview_ds(train_ds, n_rows=3, n_tokens = 500):
    check_data = train_ds.take(1)
    for text_batch, label_batch in check_data.take(1):
        text_batch = text_batch.numpy()
        label_batch = label_batch.numpy()
        
        for i in range(n_rows):
            print(f"- Text:\t {text_batch[i][:n_tokens]}")
            print(f"- Label: {label_batch[i]}")
            print()


def check_batch_size(dataset):
    # Inspect one sample batch to get the batch size
    for x_batch, y_batch in dataset.take(1):
        batch_size = x_batch.shape[0]
        print(f"The batch size is: {batch_size}")




def create_directories_from_paths(nested_dict):
    """OpenAI. (2023). ChatGPT [Large language model]. https://chat.openai.com 
    Recursively create directories for file paths in a nested dictionary.

    Parameters:
    nested_dict (dict): The nested dictionary containing file paths.
    """
    import os
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            create_directories_from_paths(value)
        elif isinstance(value, str):
            # If the value is a string, treat it as a file path and get the directory path
            directory_path = os.path.dirname(value)
            # If the directory path is not empty and the directory does not exist, create it
            if directory_path and not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Directory created: {directory_path}")








def deep_getsizeof(obj, seen=None, unit='MB', top_level=True, return_size=True):
    """
    # Function provided by OpenAI's ChatGPT
    # Date: November 1, 2023
    
    Calculate the deep size of a Python object including nested objects.

    Args:
        obj (object): The Python object whose size is to be calculated.
        seen (set, optional): A set of object ids to handle circular references. Defaults to None.
        unit (str, optional): The unit in which to return the size. 
                              Options are 'B' for Bytes, 'KB' for Kilobytes,
                              'MB' for Megabytes, 'GB' for Gigabytes. Defaults to 'B'.
        top_level (bool, optional): Whether the function is called at the top-level (not recursively).
                                    Defaults to True.
                              
    Returns:
        float: The size of the object in the unit specified.

    Example:
        >>> my_dict = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'inner_key': 'value'}}
        >>> deep_getsizeof(my_dict, unit='KB')
    """
    import sys
    size = sys.getsizeof(obj)
    
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([deep_getsizeof(v, seen, unit, False) for v in obj.values()])
        size += sum([deep_getsizeof(k, seen, unit, False) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += deep_getsizeof(obj.__dict__, seen, unit, False)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([deep_getsizeof(i, seen, unit, False) for i in obj])
    
    if top_level:
        print_and_convert_size(size, unit)
    else:
        return size


def print_and_convert_size(size, unit='B'):
    """
    # Function provided by OpenAI's ChatGPT
    # Date: November 1, 2023
    Convert and print the size into the specified unit.

    Args:
        size (float): The size in bytes.
        unit (str, optional): The unit in which to print and return the size.
                              Options are 'B' for Bytes, 'KB' for Kilobytes,
                              'MB' for Megabytes, 'GB' for Gigabytes. Defaults to 'B'.

    Returns:
        float: The size in the unit specified.
    """
    if unit == 'KB':
        size /= 1024
        print(f"{size:.3f} KB")
    elif unit == 'MB':
        size /= (1024 * 1024)
        print(f"{size:.3f} MB")
    elif unit == 'GB':
        size /= (1024 * 1024 * 1024)
        print(f"{size:.3f} GB")
    else:
        print(f"{size:.3f} B")
    return size
    

def get_filesize(fpath, unit ="MB"):
    import os
    size = os.path.getsize(fpath)
    print_and_convert_size(size,unit=unit)



#### NOT YET USED IN CURRIC
def inspect_file(fname, units='mb',verbose=False):
    """Returns a dictionary with detailed file information including:
    - File name, extension, file size, date created, date modified, etc.
    Args:
        fname (str): filepath
        units (str, optional): Units for fileszize. (Options are "kb','mb','gb'). Defaults to 'mb'.

    Returns:
        dict: dictionary with info
    """
    import time
    import os
    import pandas as pd

    ## Get file created and modified time
    modified_time = time.ctime(os.path.getmtime(fname))
    created_time = time.ctime(os.path.getctime(fname))

    ## Get file size
    raw_size = os.path.getsize(fname)
    size = get_filesize(fname,units=units, verbose=verbose)
    str_size = f"{size} {units}"

    # Get path info
    rel_path = os.path.relpath(fname)
    abs_path =  os.path.abspath(fname)
    _, ext = os.path.splitext(fname)
    basename =os.path.basename(fname)
    dirname = os.path.dirname(fname)

    file_info ={"Filepath": fname,"Name":basename, 'Created':created_time, 'Modified':modified_time,  'Size':str_size,
    'Folder':dirname,"Ext":ext, "Size (bytes)":raw_size,
    'Relative Path':rel_path,'Absolute Path':abs_path}

    return file_info



def read_and_fix_json(JSON_FILE):
    """Attempts to read in json file of records and fixes the final character
    to end with a ] if it errors.
    
    Args:
        JSON_FILE (str): filepath of JSON file
        
    Returns:
        DataFrame: the corrected data from the bad json file
    """
    import pandas as pd
    import json
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
	
	
	
	

def write_json(new_data, filename): 
    """Adapted from: https://www.geeksforgeeks.org/append-to-json-file-using-python/"""    
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

	
 
 
         
def inspect_variables(local_vars = None,sort_col='size',exclude_funcs_mods=True, top_n=10,return_df=False,always_display=True,
show_how_to_delete=False,print_names=False):
    """
    Displays a dataframe of all variables and their size in memory,
    with the largest variables at the top. 
    
    Args:
        local_vars (locals(): Must call locals()  as first argument.
        sort_col (str, optional): column to sort by. Defaults to 'size'.
        top_n (int, optional): how many vars to show. Defaults to 10.
        return_df (bool, optional): If True, return df instead of just showing df.Defaults to False.
        always_display (bool, optional): Display df even if returned. Defaults to True.
        show_how_to_delete (bool, optional): Prints out code to copy-paste into cell to del vars. Defaults to False.
        print_names (bool, optional): [description]. Defaults to False.
    
    Raises:
        Exception: if locals() not passed as first arg
    
    
    Example Usage:
    # Must pass in local variables
    >> inspect_variables(locals())
    # To see command to delete list of vars"
    >> inspect_variables(locals(),show_how_to_delete=True)
    """
    import sys
    import inspect
    import pandas as pd
    from IPython.display import display
    if local_vars is None:
        raise Exception('Must pass "locals()" in function call. i.e. inspect_variables(locals())')


    glob_vars= [k for k in globals().keys()]
    loc_vars = [k for k in local_vars.keys()]

    var_list = glob_vars+loc_vars

    var_df = pd.DataFrame(columns=['variable','size','type'])

    exclude = ['In','Out']
    var_list = [x for x in var_list if (x.startswith('_') == False) and (x not in exclude)]

    i=0
    for var in var_list:#globals().items():#locals().items():

        if var in loc_vars:
            real_var = local_vars[var]
        elif var in glob_vars:
            real_var = globals()[var]
        else:
            print(f"{var} not found.")

        var_size = sys.getsizeof(real_var)

        var_type = []
        if inspect.isfunction(real_var):
            var_type = 'function'
            if exclude_funcs_mods:
                continue
        elif inspect.ismodule(real_var):
            var_type = 'module'
            if exclude_funcs_mods:
                continue
        elif inspect.isbuiltin(real_var):
            var_type = 'builtin'
        elif inspect.isclass(real_var):
            var_type = 'class'
        else:

            var_type = real_var.__class__.__name__


        var_row = pd.Series({'variable':var,'size':var_size,'type':var_type})
        var_df.loc[i] = var_row#pd.concat([var_df,var_row],axis=0)#.join(var_row,)
        i+=1

    # if exclude_funcs_mods:
    #     var_df = var_df.loc[var_df['type'] not in ['function', 'module'] ]

    var_df.sort_values(sort_col,ascending=False,inplace=True)
    var_df.reset_index(inplace=True,drop=True)
    var_df.set_index('variable',inplace=True)
    var_df = var_df[['type','size']]

    if top_n is not None:
        var_df = var_df.iloc[:top_n]



    if always_display:
        display(var_df.style.set_caption('Current Variables by Size in Memory'))

    if show_how_to_delete:
        print('---'*15)
        print('## CODE TO DELETE MANY VARS AT ONCE:')
        show_del_me_code(called_by_inspect_vars=True)


    if print_names ==False:
        print('#[i] set `print_names=True` for var names to copy/paste.')
        print('---'*15)
    else:
        print('---'*15)
        print('Variable Names:\n')
        print_me = [f"{str(x)}" for x in var_df.index]
        print(print_me)
    
        
    if show_del_me_code == False:
        print("[i] set `show_del_me_code=True prints copy/paste var deletion code.")
        

    if return_df:
        return var_df







def column_report(df,index_col=None, sort_column='iloc', ascending=True,
                  interactive=False, return_df=False):
    """
    Displays a DataFrame summary of each column's: 
    - name, iloc, dtypes, null value count & %, # of 0's, min, max,med,mean, etc
    
    Args:
        df (DataFrame): df to report 
        index_col (column to set as index, str): Defaults to None.
        sort_column (str, optional): [description]. Defaults to 'iloc'.
        ascending (bool, optional): [description]. Defaults to True.
        as_df (bool, optional): [description]. Defaults to False.
        interactive (bool, optional): [description]. Defaults to False.
        return_df (bool, optional): [description]. Defaults to False.

    Returns:
        column_report (df): Non-styled version of displayed df report
    """
    from ipywidgets import interact
    import pandas as pd
    from IPython.display import display

    def count_col_zeros(df, columns=None):
        import pandas as pd
        import numpy as np
        # Make a list of keys for every column  (for series index)
        zeros = pd.Series(index=df.columns)
        # use all cols by default
        if columns is None:
            columns=df.columns

        # get sum of zero values for each column
        for col in columns:
            zeros[col] = np.sum( df[col].values == 0)
        return zeros


    ##
    df_report = pd.DataFrame({'.iloc[:,i]': range(len(df.columns)),
                            'column name':df.columns,
                            'dtypes':df.dtypes.astype('str'),
                            '.isna()': df.isna().sum().round(),
                            '% na':df.isna().sum().divide(df.shape[0]).mul(100).round(2),
                            '# zeros': count_col_zeros(df),
                            '# unique':df.nunique(),
                            'min':df.min(),
                            'max':df.max(),
                            'med':df.describe().loc['50%'],
                            'mean':df.mean().round(2)})#
    ## Sort by index_col
    if index_col is not None:
        hide_index=False
        if 'iloc' in index_col:
            index_col = '.iloc[:,i]'

        df_report.set_index(index_col ,inplace=True)
    else:
        hide_index=True


    ##  Sort column
    if sort_column is None:
        sort_column = '.iloc[:,i]'


    if 'iloc' in sort_column:
        sort_column = '.iloc[:,i]'

    df_report.sort_values(by =sort_column,ascending=ascending, axis=0, inplace=True)

    dfs = df_report.style.set_caption('Column Report')
    
    if hide_index:
        display(dfs.hide_index())
    else:
        display(dfs)   

    if interactive:
        @interact(column= df_report.columns,direction={'ascending':True,'descending':False})
        def sort_df(column, direction):
            return df_report.sort_values(by=column,axis=0,ascending=direction)
    if return_df:
        return df_report
        
        
        

def show_del_me_code(called_by_inspect_vars=False):
    """Prints code to copy and paste into a cell to delete vars using a list of their names.
    Companion function inspect_variables(locals(),print_names=True) will provide var names tocopy/paste """
    from pprint import pprint
    if called_by_inspect_vars==False:
        print("#[i]Call: `inspect_variables(locals(), print_names=True)` for list of var names")

    del_me = """
    del_me= []#list of variable names
    for me in del_me:
        try:
            exec(f'del {me}')
            print(f'del {me} succeeded')
        except:
            print(f'del {me} failed')
            continue
        """
    print(del_me)



def get_methods(obj,private=False):
    """
    Retrieves a list of all non-private methods (default) from inside of obj.
    - If private==False: only returns methods whose names do NOT start with a '_'
    
    Args:
        obj (object): Object to retrieve methods from.
        private (bool): Whether to retrieve private methods or public.

    Returns:
        list: the names of all of the retrieved methods.
    """
    method_list = [func for func in dir(obj) if callable(getattr(obj, func))]
    if private:
        filt_methods = list(filter(lambda x: '_' in x[0] ,method_list))
    else:
        filt_methods = list(filter(lambda x: '_' not in x[0] ,method_list))
    return  filt_methods


def get_attributes(obj,private=False):
    """
    Retrieves a list of all non-private attributes (default) from inside of obj.
    - If private==False: only returns methods whose names do NOT start with a '_'
    
    Args:
        obj (object): Object to retrieve attributes from.
        private (bool): Whether to retrieve private attributes or public.
    
    Returns:
        list: the names of all of the retrieved attributes.
    """
    method_list = [func for func in dir(obj) if not callable(getattr(obj, func))]
    if private:
        filt_methods = list(filter(lambda x: '_' in x[0] ,method_list))
    else:
        filt_methods = list(filter(lambda x: '_' not in x[0] ,method_list))
    return  filt_methods
    
    
    
def clickable_link(path,label=None):
    """Adapted from: https://www.geeksforgeeks.org/how-to-create-a-table-with-clickable-hyperlink-to-a-local-file-in-pandas/"""
    # returns the final component of a url
    # f_url = os.path.basename(path)
    if label is None:
    # convert the url into link
        return '<a href="{}">{}</a>'.format(path, path)
    else: 
        return '<a href="{}">{}</a>'.format(path, label)  
        
        
        
        

def get_or_print_filesize(fpath, unit ="MB", print_or_return='print'):
    """Get the file size as a string, converted to the requested unit(B,KB, MB, GB)

    Args:
        fpath (string): file to analyze
        unit (str, optional): unit for conversion. Defaults to "MB".
        print_or_return (str, optional): Controls if string is returned or printed. Defaults to 'print'.

    Returns:
        string: file size + units
    """
    import os
    size = os.path.getsize(fpath)
    if unit == 'KB':
        size /= 1024
    elif unit == 'MB':
        size /= (1024**2)
    elif unit == 'GB':
        size /= (1024**3)
    # else:
        # print(f"{size:.3f} B")
    formatted_size = f"{size:.3f} {unit}"
    
    if print_or_return == 'print':
        print(formatted_size)
    else: 
        return formatted_size
    