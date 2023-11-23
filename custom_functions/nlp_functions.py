"""
nlp functions from Coding Dojo Online Data Science and Machine Learning 24-Week Program.
"""


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

import spacy


def get_ngram_measures_finder(tokens=None,docs=None, ngrams=2, verbose=False,
                              get_scores_df=False, measure='raw_freq', top_n=None,
                             words_colname='Words'):
    import nltk
    if ngrams == 4:
        MeasuresClass = nltk.collocations.QuadgramAssocMeasures
        FinderClass = nltk.collocations.QuadgramCollocationFinder
        
    elif ngrams == 3: 
        MeasuresClass = nltk.collocations.TrigramAssocMeasures
        FinderClass = nltk.collocations.TrigramCollocationFinder
    else:
        MeasuresClass = nltk.collocations.BigramAssocMeasures
        FinderClass = nltk.collocations.BigramCollocationFinder

    measures = MeasuresClass()
    
    if (tokens is not None):
        finder = FinderClass.from_words(tokens)
    elif (docs is not None):
        finder = FinderClass.from_docs(docs)
    else:
        raise Exception("Must provide tokens or docs")
        

    if get_scores_df == False:
        return measures, finder
    else:
        df_ngrams = get_score_df(measures, finder, measure=measure, top_n=top_n, words_colname=words_colname)
        return df_ngrams




def get_score_df( measures,finder, measure='raw_freq', top_n=None, words_colname="Words"):
    import pandas as pd
    if measure=='pmi':
        scored_ngrams = finder.score_ngrams(measures.pmi)
    else:
        measure='raw_freq'
        scored_ngrams = finder.score_ngrams(measures.raw_freq)

    df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_",' ').title()])
    if top_n is not None:
        return df_ngrams.head(top_n)
    else:
        return df_ngrams


    

def preprocess_text(txt, nlp=None, remove_stopwords=True, remove_punct=True, use_lemmas=False,):
	"""Preprocess text into tokens/lemmas. 

	Args:
		txt (string): text to process
		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None
  									if None, it creates a default 'en_core_web_sm' pipe.
		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
		use_lemmas (bool, optional): lemmatize tokens. Defaults to False.

	Returns:
		list: list of tokens/lemmas
	"""
	if nlp is None:
		nlp = spacy.load('en_core_web_sm')

	doc = nlp(txt)

	# Saving list of the token objects for stopwords and punctuation removal
	tokens = []

	for token in doc:
		# Check if should remove stopwords and if token is stopword
		if (remove_stopwords == True) & (token.is_stop == True):
			# Continue the loop with the next token
			continue
	
		# Check if should remove punctuation and if token is punctuation
		if (remove_punct == True) & (token.is_punct == True):
			# Continue the loop with the next oken
			continue

		# Check if should remove punctuation and if token is a space
		if (remove_punct == True) & (token.is_space == True):
			# Continue the loop with the next oken
			continue
	
		# Determine final form of output list of tokens/lemmas
		if use_lemmas:
			tokens.append(token.lemma_)
		else:
			tokens.append(token.text.lower())

	return tokens





def make_custom_nlp(
	disable=["ner"],
	contractions=["don't", "can't", "couldn't", "you'd", "I'll"],
	stopwords_to_add=[],
	stopwords_to_remove=[],
    spacy_model = "en_core_web_sm"
):
	"""Returns a custom spacy nlp pipeline.
	
	Args:
		disable (list, optional): Names of pipe components to disable. Defaults to ["ner"].
		contractions (list, optional): List of contractions to add as special cases. Defaults to ["don't", "can't", "couldn't", "you'd", "I'll"].
		stopwords_to_add(list, optional): List of words to set as stopwords (word.is_stop=True)
		stopwords_to_remove(list, optional): List of words to remove from stopwords (word.is_stop=False)
			
	Returns:
		nlp pipeline: spacy pipeline with special cases and updated nlp..Default.stopwords
	"""
	# Load the English NLP model
	nlp = spacy.load(spacy_model, disable=disable)
	

	## Adding Special Cases 
	# Loop through the contractions list and add special cases
	for contraction in contractions:
		special_case = [{"ORTH": contraction}]
		nlp.tokenizer.add_special_case(contraction, special_case)

	
	## Adding stopwords
	for word in stopwords_to_add:
		# Set the is_stop attribute for the word in the vocab dict to true.
		nlp.vocab[
			word
		].is_stop = True  # this determines spacy's treatmean of the word as a stop word

		# Add the word to the list of stopwords (for easily tracking stopwords)
		nlp.Defaults.stop_words.add(word)

	
	## Removing Stopwords
	for word in stopwords_to_remove:
		
		# Ensure the words are not recognized as stopwords
		nlp.vocab[word].is_stop = False
		nlp.Defaults.stop_words.discard(word)
		
	return nlp




def custom_preprocess_text(
	txt,
	nlp=None,
	nlp_creation_fn=None,
	nlp_fn_kwargs={},  ## THESE ARE NEW SINCE BRENDA SAW
	lowercase=True,
	remove_stopwords=True,
	remove_punct=True,
	use_lemmas=False,
	disable=None,
):
	"""Preprocess text into tokens/lemmas. 
	
	Args:
		txt (string): text to process
		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
		nlp_creation_fn (_type_, optional): Function that returns an nlp pipe. Defaults to None; Only used if nlp arg is None.
		nlp_fn_kwargs (dict, optional): Keyword arguments for nlp_creation_fn. Defaults to {}.
		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
		use_lemmas (bool, optional): lemmatize tokens. Defaults to False.
		disable (list of strings, optional): named pipeline elements to disable. Defaults to None;Only used if nlp is None and nlp_creation_fn is None
	
	Returns:
		list: list of tokens/lemmas
	"""
	# If nlp is none, use nlp_creation_func to make it
	if (nlp is None) and (nlp_creation_fn is not None):
		nlp = nlp_creation_fn(**nlp_fn_kwargs)
	
	# If nlp is none,and no nlp_creation_func, make default nlp object
	elif (nlp is None) & (nlp_creation_fn is None):
		nlp = spacy.load("en_core_web_sm")

	# Create the document
	doc = nlp(txt)
	
	# Saving list of the token objects for stopwords and punctuation removal
	tokens = []
	
	for token in doc:
		# Check if should remove stopwords and if token is stopword
		if (remove_stopwords == True) & (token.is_stop == True):
			# Continue the loop with the next token
			continue
	
		# Check if should remove punctuation and if token is punctuation
		if (remove_punct == True) & (token.is_punct == True):
			# Continue the loop with the next oken
			continue

		# Check if should remove punctuation and if token is a space
		if (remove_punct == True) & (token.is_space == True):
			# Continue the loop with the next oken
			continue
	
		## Determine final form of output list of tokens/lemmas
		if use_lemmas:
			tokens.append(token.lemma_)
		elif lowercase==True:
			tokens.append(token.text.lower())
		else: 
			tokens.append(token.text)
	
	return tokens




# def batch_preprocess_texts(
# 	texts,
# 	nlp=None,
# 	remove_stopwords=True,
# 	remove_punct=True,
# 	use_lemmas=False,
# 	disable=["ner"],
# 	batch_size=50,
# 	n_process=-1,
# ):
# 	"""Efficiently preprocess a collection of texts using nlp.pipe()

# 	Args:
# 		texts (collection of strings): collection of texts to process (e.g. df['text'])
# 		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
# 		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
# 		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
# 		use_lemmas (bool, optional): lemmatize tokens. Defaults to False.
# 		disable (list of strings, optional): named pipeline elements to disable. Defaults to ["ner"]: Used with nlp.pipe(disable=disable)
# 		batch_size (int, optional): Number of texts to process in a batch. Defaults to 50.
# 		n_process (int, optional): Number of CPU processors to use. Defaults to -1 (meaning all CPU cores).

# 	Returns:
# 		list of tokens
# 	"""
# 	# from tqdm.notebook import tqdm
# 	from tqdm import tqdm

# 	if nlp is None:
# 		nlp = spacy.load("en_core_web_sm")

# 	processed_texts = []

# 	for doc in tqdm(nlp.pipe(texts, disable=disable, batch_size=batch_size, n_process=n_process)):
# 		tokens = []
# 		for token in doc:
# 			# Check if should remove stopwords and if token is stopword
# 			if (remove_stopwords == True) and (token.is_stop == True):
# 				# Continue the loop with the next token
# 				continue

# 			# Check if should remove stopwords and if token is stopword
# 			if (remove_punct == True) and (token.is_punct == True):
# 				continue

# 			## Determine final form of output list of tokens/lemmas
# 			if use_lemmas:
# 				tokens.append(token.lemma_)
# 			else:
# 				tokens.append(token.text.lower())

# 		processed_texts.append(tokens)
# 	return processed_texts



# def custom_batch_preprocess_texts(
# 	texts,
# 	nlp=None,
# 	remove_stopwords=True,
# 	remove_punct=True,
# 	use_lemmas=False,
# 	disable=["ner"],
# 	batch_size=50,
# 	n_process=-1,
# 	nlp_creation_fn=None,
# 	nlp_fn_kwargs={},
# ):
# 	"""Efficiently preprocess a collection of texts using nlp.pipe()
	
# 	Args:
# 		texts (collection of strings): collection of texts to process (e.g. df['text'])
# 		nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
# 		remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
# 		remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
# 		use_lemmas (bool, optional): lemmatize tokens. Defaults to False.
# 		disable (list of strings, optional): named pipeline elements to disable. Defaults to ["ner"]; Used with nlp.pipe(disable=disable)
# 		batch_size (int, optional): Number of texts to process in a batch. Defaults to 50.
# 		n_process (int, optional): Number of CPU processors to use. Defaults to -1 (meaning all CPU cores).
# 		nlp_creation_fn (function, optional): Function that returns an nlp pipe. Defaults to None; Only used if nlp arg is None.
# 		nlp_fn_kwargs (dict, optional): Keyword arguments for nlp_creation_fn. Defaults to {}.
# 	Returns:
# 		list of tokens
# 	"""
# 	from  tqdm import tqdm
# 	if (nlp is None) and (nlp_creation_fn is not None):
# 		nlp = nlp_creation_fn(**nlp_fn_kwargs)

# 	elif (nlp is None) & (nlp_creation_fn is None):
# 		nlp = spacy.load("en_core_web_sm")


# 	processed_texts = []

# 	for doc in tqdm(nlp.pipe(texts, disable=disable, batch_size=batch_size, n_process=n_process)):
# 		tokens = []
# 		for token in doc:
# 			# Check if should remove stopwords and if token is stopword
# 			if (remove_stopwords == True) and (token.is_stop == True):
# 				# Continue the loop with the next token
# 				continue

# 			# Check if should remove stopwords and if token is stopword
# 			if (remove_punct == True) and (token.is_punct == True):
# 				continue

# 			## Determine final form of output list of tokens/lemmas
# 			if use_lemmas:
# 				tokens.append(token.lemma_)
# 			else:
# 				tokens.append(token.text.lower())

# 		processed_texts.append(tokens)
# 	return processed_texts



#### ADDING ANN FUNCTIONS HERE

      
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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





import tensorflow as tf
@tf.function
def extract_text(x,y):
    return x



def standardize_remove_stopwords(input_text,strip_punctuation=True, stopwords='english', 
                          add_stopwords=[], remove_stopwords=[]):
    """ChatGPT Gave start tf.strings.lower and tf.strings.regecx_replace code 
    if stopwords = None, only words pass in as add_stopwords will be included
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    if stopwords == 'english':
        custom_stopwords = sorted(list(ENGLISH_STOP_WORDS))
    else:
        custom_stopwords  = []

    ## Add/remove stopwords
    for word in add_stopwords:
        custom_stopwords.append(word)
    for word in remove_stopwords:
        custom_stopwords.remove(word)

    ## Add punctuation to stopwords
    if strip_punctuation:
        from string import punctuation
        for char in list(punctuation):
            custom_stopwords.append(f"\{char}")#list(punctuation))

    
    lowercase_text = tf.strings.lower(input_text)
    return tf.strings.regex_replace(lowercase_text, f'({"|".join(custom_stopwords)})', '')



def make_text_vectorization_layer(train_ds,  max_tokens=None, 
                                  split='whitespace',
                                  standardize="lower_and_strip_punctuation",
                                  output_mode="int",
                                  output_sequence_length=None,
                                  ngrams=None, pad_to_max_tokens=False,
                                  # return_lookups='i'
                                  verbose=True,
                                  **kwargs,
                                 ):
    from pprint import pprint
    # ADMIN/temp
    ## Setting default params for those I decided to not include in this funcion
    # anything below can be overridden by adding kwargs to function call)
    default_kwargs = dict(vocabulary=None,
                                  idf_weights=None,
                                  sparse=False,
                                  ragged=False,
                                  encoding='utf-8')
    default_kwargs.update(kwargs)

    ## CRETING THE TextVectorization LAYER FOR TFIDF!
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=standardize, 
        output_mode=output_mode,
        output_sequence_length=output_sequence_length,**default_kwargs
    )

    ## TRAINING THE VECTORIZATION LAYER
    # Get jsut the training ds texts
    if isinstance(train_ds, (np.ndarray, list, tuple, pd.Series)):
        ds_texts = train_ds
    else:
        # if isinstance(tf.data.Dataset):
        try:
            ds_texts = train_ds.map(extract_text)#train_ds.map(lambda x, y: x )
        except:
            ds_texts = train_ds#.map(lambda x: x )
            
    # Fit the layer on the training texts
    text_vectorizer.adapt(ds_texts)
    
    
    if verbose:
        ## Print the params
        print( "\ntf.keras.layers.TextVectorization(" )
        config = text_vectorizer.get_config()
        pprint(config,indent=4)
        print(")")
               
    #### SAVING VOCAB FOR LATER

    # Getting list of vocab 
    vocab = text_vectorizer.get_vocabulary()
    ## Save dixtionaries to lookup words from ints and vice versa
    int_to_str  = {idx:word for idx,word in enumerate(vocab)}
    # str_to_int = {word:idx for idx,word in int_to_str.items()}
    
        # type(vocab), len(vocab), vocab[:6]
    return text_vectorizer, int_to_str#, str_to_int



def train_val_test_datasets(X,y,BATCH_SIZE=32, train_size = 0.7, 
                    val_size = 0.1, SEED = 321,shuffle_train=True, verbose=False,
                            show_class_balance=True, cache_prefetch=True,
                            prefetch_buffer='AUTOTUNE'
                           ):
    
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Splitting val_test into 3 splits
    # Set the ratio of the train, validation, test split
    split_train = train_size
    split_val =  val_size
    split_test =  1 -( split_train + split_val )
    
    # Calculate the number of batches for training and validation data 
    n_train_samples =  int(len(ds) * split_train)
    n_val_samples = int(len(ds) * split_val)
    n_test_samples = len(ds) -(n_train_samples + n_val_samples)

    if verbose:
        import math
        print(f"[i] Creating datasets with batch_size={BATCH_SIZE}:")
        n_train_batches = math.ceil(n_train_samples/BATCH_SIZE)
        n_val_batches = math.ceil(n_test_samples/BATCH_SIZE)
        n_test_batches = math.ceil(n_val_samples/BATCH_SIZE)
        
        print(f"    - train:\t{n_train_samples} samples \t({n_train_batches} batches)")
        print(f"    - val:  \t{n_val_samples} samples \t({ n_val_batches} batches)")
        print(f"    - test: \t{n_test_samples} samples \t({n_test_batches} batches)\n")

    
    # Use .take to slice out the number of batches 
    ds = ds.shuffle(buffer_size=len(ds),reshuffle_each_iteration=False) # Adding shuffling once)
    # Making Train ds
    train_ds = ds.take(n_train_samples).batch(batch_size=BATCH_SIZE)
    
    if shuffle_train:        
        print("\n[!] shuffle_train default value has changed to True.\n")
        train_ds = train_ds.shuffle(buffer_size=len(train_ds), seed=SEED)#.batch(batch_size=BATCH_SIZE)

    # Making Val Ds
    # Skipover the training batches
    val_ds = ds.skip(n_train_samples).take(n_val_samples).batch(batch_size=BATCH_SIZE)

    # Making Test Ds
    # Skip over all of the training + validation batches
    test_ds = ds.skip(n_train_samples + n_val_samples).batch(batch_size=BATCH_SIZE)


    ## Add prefetching
    if cache_prefetch==True:
        if prefetch_buffer == 'AUTOTUNE':
            BUFFER_SIZE = tf.data.AUTOTUNE
        else:
            BUFFER_SIZE = prefetch_buffer
            
        train_ds = train_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
        val_ds = val_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
        test_ds = test_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
        
        if verbose:
            print(f'[i] Using cache and prefetch with buffer_size = {BUFFER_SIZE}.')


    
    ## SHOW CLASS BALANCE
    if verbose & show_class_balance:
        print('\n[i] Previewing Class Balance per Split:')
        for split_name, data in [('train',train_ds),('val',val_ds), ('test',test_ds)]:

            
            labels = get_labels(data, counts=False, return_ohe_labels=False)
            print_labels = labels.value_counts(dropna=False).sort_index().to_dict()

                # counts = labels.sum(axis=1)
                # print_labels = {i:count for i,count in enumerate(counts)}
                
            print(f"    - {split_name}:\t",print_labels )

    return train_ds, val_ds, test_ds


## Checking class balance in each split
def get_labels(ds, counts=False, return_ohe_labels = False):
    """Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
    Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    """
    if counts==True:
        raise Exception("Counts has been eliminated. run .value_counts() on output series instead.")
    
    y_true = []    
    # Loop through the dataset as a numpy iterator
    for data, labels in ds.as_numpy_iterator():
        y_true.extend(labels)
        
    ## Convert the lists to arrays
    y_true =  np.array(y_true)

    ## If 2D labels, get the argmax
    if return_ohe_labels:
        return y_true
    else:        
        if y_true.ndim > 1:
            return pd.Series(y_true.argmax(axis=1))
            
        else:
            return pd.Series(y_true)

    # if counts==True:
    #     return y_true.value_counts(dropna=False)
    # else:
    #     return y_true







##################### FUNCTIONS FROM IML#####################
"""Functions from Intermediate Machine Learning Wk3-4 Lessons"""

def plot_history(history,figsize=(6,8)):
    # Get a unique list of metrics 
    all_metrics = np.unique([k.replace('val_','') for k in history.history.keys()])

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
        axes[i].plot(epochs, score, label=metric, marker='.')
        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}",marker='.')
        except:
            pass

        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch",ylabel=metric)

    # Adjust subplots and show
    fig.tight_layout()
    plt.show()
    return fig




# def evaluate_classification(model, X_train=None, y_train=None, X_test=None, y_test=None,
#                          figsize=(6,4), normalize='true', output_dict = False,
#                             cmap_train='Blues', cmap_test="Reds",colorbar=False,label=''):

#     ## Adding a Print Header
#     print("\n"+'='*70)
#     print(f'- Evaluating Model:    {label}')
#     print('='*70)

    

#     if ( X_train is not None) & (y_train is not None):
#         # Get predictions for training data
#         y_train_pred = model.predict(X_train)
        
#         # Call the helper function to obtain regression metrics for training data
#         results_train = classification_metrics(y_train, y_train_pred, #verbose = verbose,
#                                          output_dict=True, figsize=figsize,
#                                              colorbar=colorbar, cmap=cmap_train,
#                                          label='Training Data')
#         print()

#     if ( X_test is not None) & (y_test is not None):
#         # Get predictions for test data
#         y_test_pred = model.predict(X_test)
#         # Call the helper function to obtain regression metrics for test data
#         results_test = classification_metrics(y_test, y_test_pred, #verbose = verbose,
#                                       output_dict=True,figsize=figsize,
#                                              colorbar=colorbar, cmap=cmap_test,
#                                         label='Test Data' )
#     if output_dict == True:
#         # Store results in a dataframe if ouput_frame is True
#         results_dict = {'train':results_train,
#                     'test': results_test}
#         return results_dict



### FINAL FROM FLEXIBILE EVAL FUNCTIONS LESSON
   
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def classification_metrics(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f"):
    """Classification metrics function from Intro to Machine Learning"""
    # Get the classification report
    report = classification_report(y_true, y_pred)
    ## Print header and report
    header = "-"*70
    print(header, f" Classification Metrics: {label}", header, sep='\n')
    print(report)
    
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # create a confusion matrix  of raw counts
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                normalize=None, cmap='gist_gray_r', values_format="d", colorbar=colorbar,
                ax = axes[0],);
    axes[0].set_title("Raw Counts")
    
    # create a confusion matrix with the test data
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                normalize=normalize, cmap=cmap, values_format=values_format, colorbar=colorbar,
                ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()
    
    # Return dictionary of classification_report
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict
def classification_metrics(
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
    results_folder="Models/results/",
    model_prefix="ml",
    verbose=True,
    savefig_kws={},
):
    """Classification metrics function from Advanced Machine Learning
    - Saves the classification report and figure to file for easy use"""
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

        ## Save classification report
        report_fname = results_folder + f"{model_prefix}-class-report-{label}.txt"
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

# def get_true_pred_labels(model,ds):
#     """Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
#     Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
#     """
#     y_true = []
#     y_pred_probs = []
    
#     # Loop through the dataset as a numpy iterator
#     for images, labels in ds.as_numpy_iterator():
        
#         # Get prediction with batch_size=1
#         y_probs = model.predict(images, batch_size=1, verbose=0)

#         # Combine previous labels/preds with new labels/preds
#         y_true.extend(labels)
#         y_pred_probs.extend(y_probs)

#     ## Convert the lists to arrays
#     y_true = np.array(y_true)
#     y_pred_probs = np.array(y_pred_probs)
    
#     return y_true, y_pred_probs
    


# def evaluate_classification_network(model, 
#                                     X_train=None, y_train=None, 
#                                     X_test=None, y_test=None,
#                                     history=None, history_figsize=(6,6),
#                                     figsize=(6,4), normalize='true',
#                                     output_dict = False,
#                                     cmap_train='Blues',
#                                     cmap_test="Reds",
#                                     values_format=".2f", 
#                                     colorbar=False, 
#                                    return_results=False,
#                                    label=""):
#     """Evaluates a neural network classification task using either
#     separate X and y arrays or a tensorflow Dataset
    
#     Data Args:
#         X_train (array, or Dataset)
#         y_train (array, or None if using a Dataset
#         X_test (array, or Dataset)
#         y_test (array, or None if using a Dataset)
#         history (history object)
#         """
#     # Plot history, if provided
#     if history is not None:
#         plot_history(history, figsize=history_figsize)

#     ## Adding a Print Header
#     print("\n"+'='*70)
#     print(f'- Evaluating Network:    {label}')
#     print('='*70)

    
#     ## TRAINING DATA EVALUATION
#     # check if X_train was provided
#     if X_train is not None:
        
#         ## Check if X_train is a dataset
#         if hasattr(X_train,'map'):
#             # If it IS a Datset:
#             # extract y_train and y_train_pred with helper function
#             y_train, y_train_pred = get_true_pred_labels(model, X_train)
#         else:
#             # Get predictions for training data
#             y_train_pred = model.predict(X_train)

#         ## Pass both y-vars through helper compatibility function
#         y_train = convert_y_to_sklearn_classes(y_train)
#         y_train_pred = convert_y_to_sklearn_classes(y_train_pred)

        
#         # Call the helper function to obtain regression metrics for training data
#         results_train = classification_metrics(y_train, y_train_pred, 
#                                          output_dict=True, figsize=figsize,
#                                              colorbar=colorbar, cmap=cmap_train,
#                                                values_format=values_format,
#                                          label='Training Data')
        
#         ## Run model.evaluate         
#         print("\n- Evaluating Training Data:")
#         print(model.evaluate(X_train, return_dict=True))
    
#     # If no X_train, then save empty list for results_train
#     else:
#         results_train = []


#     ## TEST DATA EVALUATION
#     # check if X_test was provided
#     if X_test is not None:
#         ## Check if X_train is a dataset
#         if hasattr(X_test,'map'):
#             # If it IS a Datset:
#             # extract y_train and y_train_pred with helper function
#             y_test, y_test_pred = get_true_pred_labels(model, X_test)
#         else:
#             # Get predictions for training data
#             y_test_pred = model.predict(X_test)

#         ## Pass both y-vars through helper compatibility function
#         y_test = convert_y_to_sklearn_classes(y_test)
#         y_test_pred = convert_y_to_sklearn_classes(y_test_pred)
        
#         # Call the helper function to obtain regression metrics for training data
#         results_test = classification_metrics(y_test, y_test_pred, 
#                                          output_dict=True, figsize=figsize,
#                                              colorbar=colorbar, cmap=cmap_test,
#                                               values_format=values_format,
#                                          label='Test Data')
        
#         ## Run model.evaluate         
#         print("\n- Evaluating Test Data:")
#         print(model.evaluate(X_test, return_dict=True))
      
#     # If no X_test, then save empty list for results_test
#     else:
#         results_test = []
    
#     if return_results == True:
#         return {'train':results_train, 'test': results_test}
        
# def evaluate_classification(model, X_train=None, y_train=None, X_test=None, y_test=None,
#                             figsize=(6,4), normalize='true', output_dict = False,
#                             cmap_train='Blues', cmap_test="Reds",colorbar=False,label='',
#                             # New Args - Saving Results
#                             target_names=None, save_results=False,  verbose=True, report_label=None,
#                             results_folder="Models/results/", model_prefix="ml", savefig_kws={},
#                             # New Args - Saving Model
#                             save_model=False, save_data=False,
#                            ):
#     """Updated Verson of Intro to ML's evaluate_classification function"""

#     ## Adding a Print Header
#     print("\n"+'='*70)
#     print(f'- Evaluating Model:    {label}')
#     print('='*70)

#     ## Changing value of output dict if saving results
#     if save_results == True:
#         output_dict=False

#     if ( X_train is not None) & (y_train is not None):
        
#         # make the final label names for classification report's header
#         if report_label is not None:
#             report_label_ = report_label + " - Training Data"
#         else: 
#             report_label_ = 'Training Data'

        
#         # Get predictions for training data
#         y_train_pred = model.predict(X_train)
        
#         # Call the helper function to obtain regression metrics for training data
#         fpaths_or_results_train = classification_metrics(
#             y_train,
#             y_train_pred,  
#             output_dict=output_dict,
#             figsize=figsize,
#             colorbar=colorbar,
#             cmap=cmap_train,
#             target_names=target_names,
#             save_results=save_results,
#             verbose=verbose,
#             label=report_label_,
#             results_folder=results_folder,
#             model_prefix=model_prefix,
#             savefig_kws=savefig_kws,
#         )

#         print()

#     if ( X_test is not None) & (y_test is not None):
#         # make the final label names for classification report's header
#         if report_label is not None:
#             report_label_ = report_label + " - Test Data"
#         else: 
#             report_label_ = 'Test Data'

#         # Get predictions for test data
#         y_test_pred = model.predict(X_test)
        
#        # Call the helper function to obtain regression metrics for training data
#         fpaths_or_results_test = classification_metrics(
#             y_test,
#             y_test_pred,  
#             output_dict=output_dict,
#             figsize=figsize,
#             colorbar=colorbar,
#             cmap=cmap_test,
#             label=report_label_,
#             target_names=target_names,
#             save_results=save_results,
#             verbose=verbose,
#             results_folder=results_folder,
#             model_prefix=model_prefix,
#             savefig_kws=savefig_kws,
#         )

#     # Save a joblib file
#     if save_model == True:
#         import joblib
#         to_save = {'model':model}
    
#         if save_data == True:
#             vars = {'X_train':X_train,'y_train': y_train,'X_train': X_train,'y_train': y_train}
#             for name, var in vars.items():
#                 if var is not None:
#                     to_save[name] = var

#         # Save joblib
#         fpath_joblib = results_folder+f"model-{model_prefix}.joblib"
#         joblib.dump(to_save, fpath_joblib)
    
            
        
        
#     ## If either output_dict or save_results
#     if (save_results==True) | (output_dict==True):
#         # Store results in a dict if ouput_frame or save_results is True
#         results_dict = {'train':fpaths_or_results_train,
#                     'test': fpaths_or_results_test}
#         if save_model == True:
#             results_dict['model-joblib'] = fpath_joblib
#         return results_dict
        

# def evaluate_classification_network(
#     model,
#     X_train=None,
#     y_train=None,
#     X_test=None,
#     y_test=None,
#     history=None,
#     history_figsize=(6, 6),
#     figsize=(6, 4),
#     normalize="true",
#     output_dict=False,
#     cmap_train="Blues",
#     cmap_test="Reds",
#     values_format=".2f",
#     colorbar=False,
#     # return_results=False,
#     label="",
#     # New Args
#     target_names=None,
#     save_results=False,
#     verbose=True,
#     report_label=None,
#     results_folder="Models/results/",
#     model_prefix="nn",
#     savefig_kws={},
#     save_model=False, 
#     save_data=False,
#     model_save_fmt='tf',
#     model_save_kws = {}
# ):
#     """Evaluates a neural network classification task using either
#     separate X and y arrays or a tensorflow Dataset

#     Data Args:
#         X_train (array, or Dataset)
#         y_train (array, or None if using a Dataset
#         X_test (array, or Dataset)
#         y_test (array, or None if using a Dataset)
#         history (history object)
#     """

#     filepaths = {}
#     # Plot history, if provided
#     if history is not None:
#         fig = plot_history(history, figsize=history_figsize)

#         ## New Code For Saving Results
#         if save_results == True:
#             import os
#             # Create the results foder
#             os.makedirs(results_folder, exist_ok=True)

#             ## Save classification report
#             history_fname = results_folder + f"{model_prefix}-history.png"
#             filepaths['history'] = history_fname
#             ## Save figure
#             fig.savefig(
#                 history_fname, transparent=False, bbox_inches="tight", **savefig_kws
#             )
#             if verbose:
#                 print(f"- Model History saved as {history_fname}")

#     ## Adding a Print Header
#     print("\n" + "=" * 70)
#     print(f"- Evaluating Network:    {label}")
#     print("=" * 70)

#     ## Changing value of output dict if saving results
#     if save_results == True:
#         output_dict = False

#     ## TRAINING DATA EVALUATION
#     # check if X_train was provided
#     if X_train is not None:
#         # make the final label names for classification report's header
#         if report_label is not None:
#             report_label_ = report_label + " - Training Data"
#         else:
#             report_label_ = "Training Data"

#         ## Check if X_train is a dataset
#         if hasattr(X_train, "map"):
#             # If it IS a Datset:
#             # extract y_train and y_train_pred with helper function
#             y_train, y_train_pred = get_true_pred_labels(model, X_train)
#         else:
#             # Get predictions for training data
#             y_train_pred = model.predict(X_train)

#         ## Pass both y-vars through helper compatibility function
#         y_train = convert_y_to_sklearn_classes(y_train)
#         y_train_pred = convert_y_to_sklearn_classes(y_train_pred)

#         # Call the helper function to obtain regression metrics for training data
#         fpaths_or_results_train = classification_metrics(
#             y_train,
#             y_train_pred,
#             output_dict=output_dict,
#             figsize=figsize,
#             colorbar=colorbar,
#             cmap=cmap_train,
#             values_format=values_format,
#             label=report_label_,
#             target_names=target_names,
#             save_results=save_results,
#             verbose=verbose,
#             results_folder=results_folder,
#             model_prefix=model_prefix,
#             savefig_kws=savefig_kws,
#         )

#         ## Run model.evaluate
#         print("\n- Evaluating Training Data:")
#         print(model.evaluate(X_train, return_dict=True))

#     # If no X_train, then save empty list for results_train
#     else:
#         fpaths_or_results_train = []

#     ## TEST DATA EVALUATION
#     # check if X_test was provided
#     if X_test is not None:
#         # make the final label names for classification report's header
#         if report_label is not None:
#             report_label_ = report_label + " - Test Data"
#         else:
#             report_label_ = "Test Data"

#         ## Check if X_train is a dataset
#         if hasattr(X_test, "map"):
#             # If it IS a Datset:
#             # extract y_train and y_train_pred with helper function
#             y_test, y_test_pred = get_true_pred_labels(model, X_test)
#         else:
#             # Get predictions for training data
#             y_test_pred = model.predict(X_test)

#         ## Pass both y-vars through helper compatibility function
#         y_test = convert_y_to_sklearn_classes(y_test)
#         y_test_pred = convert_y_to_sklearn_classes(y_test_pred)

#         # Call the helper function to obtain regression metrics for training data
#         fpaths_or_results_test = classification_metrics(
#             y_test,
#             y_test_pred,
#             output_dict=output_dict,
#             figsize=figsize,
#             colorbar=colorbar,
#             cmap=cmap_test,
#             label=report_label_,
#             target_names=target_names,
#             save_results=save_results,
#             verbose=verbose,
#             results_folder=results_folder,
#             model_prefix=model_prefix,
#             savefig_kws=savefig_kws,
#         )

#         ## Run model.evaluate
#         print("\n- Evaluating Test Data:")
#         print(model.evaluate(X_test, return_dict=True))
#     else:
#         fpaths_or_results_test = []


#     if save_model:
#         model_fpath = results_folder+f"model-{model_prefix}" 
#         model.save(model_fpath, **model_save_kws ,save_format=model_save_fmt)

#         if save_data:
#             # Saving Tensorflow dataset to tfrecord
#             if X_test is not None:
#                 fname_test_ds =results_folder+f"model-{model_prefix}-test-ds" # test_data_folder+"test-ds"#.tfrecord"
#                 X_test.save(path=fname_test_ds,)
#             else: 
#                 raise Exception("[!] save_data=True, but X_test = None!")
                
            

    
#     ## If either output_dict or save_results
#     if (save_results == True) | (output_dict == True):
#         # Store results in a dict if ouput_frame or save_results is True
#         results_dict = {**filepaths,
#             "train": fpaths_or_results_train,
#             "test": fpaths_or_results_test,
#         }

#         if save_model == True:
#             results_dict['model'] = model_fpath

#             if save_data == True:
#                 results_dict['test-ds'] = fname_test_ds
            
#         return results_dict


