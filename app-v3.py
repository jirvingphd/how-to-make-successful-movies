import streamlit as st


# https://docs.streamlit.io/library/api-reference/widgets/st.button

# if st.button("Show Cheatsheet",key='cheatsheet'):
	# st.markdown("Reference/Streamlit Cheatsheet.md")

######### PY FILE VERSION:

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image


# Changing the Layout
st.set_page_config( layout="wide")


# FIX IMPORT OF NLP_FUNCTIONS

import sys, os
# sys.path.append(os.path.abspath("../../NLP/"))
import custom_functions as fn
# import nlp_functions as fn

# st.write(os.getcwd())
BASE_DIR = "./"


# Get Fpaths
@st.cache_data
def get_app_FPATHS(fpath='config/filepaths.json'):
	import json
	with open(fpath ) as f:
		return json.load(f)


# Get lookup dict
@st.cache_data
def load_target_lookup(fpath_target_map):
	import json
	with open(fpath_target_map) as f:
		lookup_dict = json.load(f)
	## Temp: move this to before saving the lookup
	final_dict = lookup_dict['lookup_names']
	final_dict = {int(k):str(v) for k,v in final_dict.items()}
	return final_dict


# Ml Model Results

# @st.cache_data
def load_ml_model_results(fpath_dict=None,report_fname=None, conf_mat_fname=None, ):
	
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
	
	return {'classification_report':report, 'confusion_matrix':conf_mat}



## Load ML Model
@st.cache_resource
def load_ml_model(fpath, model_only=True):
	import joblib 
	loaded =  joblib.load(fpath)
	if 'model' in loaded:
		return loaded['model']
	else:
		return loaded

def predict_decode(X_to_pred, clf_pipe,lookup_dict, label_format="{x} stars", return_index=True):
    
    if isinstance(X_to_pred, str):
        
        X = [X_to_pred]
    else:
        X = X_to_pred
    
    pred = clf_pipe.predict(X)
    # Decode label
    label = lookup_dict[pred[0]]
    final_label = label_format.format(x=str(label))
    
    if return_index==False:
        return final_label
    else:
        return final_label, pred




# @st.cache_data
def load_nn_model_results(fpath_dict=None,report_fname=None, conf_mat_fname=None,history_fname =None ):
	
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

	# start dict to return
	results =  {'classification_report':report, 'confusion_matrix': conf_mat} 


	if history_fname is not None:
		history =  Image.open(history_fname)
		results['history'] = history
	
	return results

@st.cache_resource
def load_nn_model(fpath_load_deep_model):
	return  tf.keras.models.load_model(fpath_load_deep_model)



def predict_decode_deep(X_to_pred, network,lookup_dict, label_format="{x} stars", 
                       return_index=True):
    
    if isinstance(X_to_pred, str):
        
        X = [X_to_pred]
    else:
        X = X_to_pred
    
    pred_probs = network.predict(X)

    pred = fn.convert_y_to_sklearn_classes(pred_probs)
    # Decode label
    label = lookup_dict[pred[0]]
    final_label = label_format.format(x=str(label))
    
    if return_index==False:
        return final_label
    else:
        return final_label, pred


# @st.cache_data
def show_results_ml(results_ml):
	# if 'history' in results_ml:
		# st.image(results_ml['history'])
	st.text(results_ml['classification_report'])
	st.image(results_ml['confusion_matrix'])
   
# @st.cache_data   
def show_results_nn(results_nn):
	st.text(results_nn['classification_report'])
	st.image(results_nn['confusion_matrix'])
	if 'history' in results_nn:
		st.image(results_nn['history'])

	
	
import nltk

# @st.cache_data
def get_ngram_measures_finder(tokens=None,docs=None, ngrams=2, verbose=False,
							  get_scores_df=False, measure='raw_freq', top_n=None,
							 words_colname='Words'):
	
	if ngrams ==4:
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
	
	
## ADDING LIME
from lime.lime_text import LimeTextExplainer
@st.cache_resource
def get_explainer(classes =['1 Stars','3 Stars','5 Stars'],**lime_kws):
	lime_explainer = LimeTextExplainer(class_names=classes, **lime_kws)
	return lime_explainer
lime_explainer = get_explainer()


	
######## APP CODE ############
FPATHS = get_app_FPATHS()
FPATHS_ml_model = FPATHS['models']['ml']['bayes']['results']
# FPATHS_nn_model = FPATHS['models']['nn']['LSTM']['results']
lookup = load_target_lookup(FPATHS['metadata']['target_lookup'])

# st.write(FPATHS_ml_model)

clf_pipe = load_ml_model(FPATHS['models']['ml']['bayes']['saved_model'])
results_ml = load_ml_model_results(FPATHS_ml_model['test'])
# network = load_nn_model(FPATHS_nn_model['model'])
# nn_results = load_nn_model_results(FPATHS_nn_model['test'],history_fname=FPATHS_nn_model['history'])



## VISIBLE APP COMPONENTS START HERE
st.title("Example Streamlit App - Getting NLP Predictions")
st.subheader("Predicting Yelp Restaurant Reviews")

st.image(FPATHS['images']['banner'],#"Images/dalle-yelp-banner-1.png",
         width=800,)
st.divider()

# st.header("Modeling")
## Load files and models

## VISIBLE APP COMPONENTS CONTINUE HERE
st.header("Get Model Predictions")

X_to_pred = st.text_input("### Enter text to predict here:", value="I just hated everything!")


# @st.cache_resource
def explain_instance(explainer, X_to_pred,predict_func, labels=(0,2) ):
	explanation = explainer.explain_instance(X_to_pred, predict_func,labels=labels)
	return explanation.as_html()

# st.markdown("> Predict & Explain:")
get_any_preds = st.button("Get Predictions for selected models:")

get_pred_ml = st.checkbox("Machine Learning Model",value=True)
get_pred_nn = False
# get_pred_nn = st.checkbox("Neural Network", value=True)


# check_explain_preds  = st.checkbox("Explain predictions with Lime",value=False)
if (get_pred_ml) & (get_any_preds):
	st.markdown(f"> #### The ML Model predicted:")
	with st.spinner("Getting Predictions..."):
		# st.write(f"[i] Input Text: '{X_to_pred}' ")
		pred, label_index_ml = predict_decode(X_to_pred, lookup_dict=lookup,clf_pipe=clf_pipe)
		
		st.markdown(f"### \t _{pred}_")

		explanation_ml = explain_instance(lime_explainer, X_to_pred, clf_pipe.predict_proba,labels=label_index_ml )#lime_explainer.explain_instance(X_to_pred, clf_pipe.predict_proba,labels=label_index_ml)
		components.html(explanation_ml)
else: 
    st.empty()


if get_pred_nn & get_any_preds:
	st.markdown(f"> #### The Neural Network predicted:")
	with st.spinner("Getting Predictions..."):
		pred_network, label_index_nn = predict_decode_deep(X_to_pred, network=network , lookup_dict=lookup, )
		# st.markdown("**From the Deep Model:**")
		# st.write(f"Prediction:\t{pred_network}!")
		st.markdown(f"### \t _{pred_network}_")
		explanation_nn = explain_instance(lime_explainer, X_to_pred, network.predict,labels=label_index_nn )#lime_explainer.explain_instance(X_to_pred, clf_pipe.predict_proba,labels=label_index_ml)
		# explanation_nn = lime_explainer.explain_instance(X_to_pred, network.predict, labels=label_index_nn)
		components.html(explanation_nn)#.as_html())
else:
	st.empty()	

st.divider()

st.header("Model Performance")

button_show_results_ml = st.checkbox("Show ML Model Results", value=False)
button_show_results_nn = st.checkbox("Show Neural Network Results", value=False)

if button_show_results_ml:
	## MACHINE LEARNING MODEL
	st.markdown('#### Machine Learning Model:')
	show_results_ml(results_ml)
	st.divider()
else:
	st.empty()

# if button_show_results_nn & button_show_results_ml:
    
 
if button_show_results_nn:
	## Load Neural Network
	st.markdown('#### Neural Network Sequence Model:')
	show_results_nn(nn_results)
else:
    st.empty()

# button_show_nn_results = st.checkbox("Neural Newtwork Results",value=True)
# if button_show_nn_results:
	

st.divider()




st.header("EDA: Explore 1-star vs 5-star Reviews")



# Load Groups Dict
@st.cache_data
def load_groups_joblib(fpath):
	import joblib
	return joblib.load(fpath)

GROUPS = load_groups_joblib(FPATHS['data']['raw']['groups-dict'])







## Add creating ngrams
st.subheader('N-Grams')


ngrams = st.radio('n-grams', [2,3,4],horizontal=True,index=1)
top_n = st.selectbox('Compare Top # Ngrams',[10,15,20,25],index=2)

## Test Code for Making New Trigrams for both groups

@st.cache_data
def show_group_ngrams(GROUPS, col='tokens', ngrams=3, top_n=20):

    group_names = list(GROUPS.keys())
    # Set kws
    ngram_kws = dict(ngrams=ngrams,  get_scores_df=True, top_n=top_n)

    ngram_df_list = []
    for group in group_names:
        tokens = GROUPS[group][col]
        ngram_df = fn.nlp.get_ngram_measures_finder(tokens=tokens, words_colname= f"{group} Review Words" ,get_scores_df=True)
        ngram_df_list.append(ngram_df)
    
    # df_ngrams_1star = fn.nlp.get_ngram_measures_finder(tokens=GROUPS[1][col], words_colname=f"1 Star n-grams", **ngram_kws)
    # df_ngrams_1star.columns =[ "1 Star n-grams", "1 Star-Freq"]
                              
    # df_ngrams_5star= get_ngram_measures_finder(tokens=GROUPS[5][col], words_colname=f"5 Star n-grams", **ngram_kws)
    # df_ngrams_5star.columns =[ "5 Star n-grams", "5 Star-Freq"]
    
    combined_ngrams = pd.concat(ngram_df_list,axis=1)
    return combined_ngrams

show_group_ngrams(GROUPS)
# def show_group_ngrams(GROUPS, col='tokens_nostop-combined-list', ngrams=3, top_n=20):
# 	ngram_kws = dict(ngrams=ngrams,  get_scores_df=True, top_n=top_n)
	
# 	df_ngrams_1star = get_ngram_measures_finder(tokens=GROUPS[1][col], words_colname=f"1 Star n-grams", **ngram_kws)
# 	df_ngrams_1star.columns =[ "1 Star n-grams", "1 Star-Freq"]
							  
# 	df_ngrams_5star= get_ngram_measures_finder(tokens=GROUPS[5][col], words_colname=f"5 Star n-grams", **ngram_kws)
# 	df_ngrams_5star.columns =[ "5 Star n-grams", "5 Star-Freq"]
	
# 	combined_ngrams = pd.concat([df_ngrams_1star,df_ngrams_5star],axis=1)
# 	return combined_ngrams

ngrams_df = show_group_ngrams(GROUPS=GROUPS,ngrams=ngrams,top_n=top_n)
st.dataframe(ngrams_df, hide_index=True, width=1000, height=None)
st.divider()

# st.divider()
## Add Displaying Wordclouds
st.subheader('WordClouds')
img = Image.open(FPATHS['eda']['wordclouds'])
st.image(img,width=1200)
st.divider()


st.subheader("ScatterText:")
@st.cache_data
def load_scattertext(fpath):
	with open(fpath) as f:
		explorer = f.read()
		return explorer


checkbox_scatter = st.checkbox("Show Scattertext Explorer",value=True)
if checkbox_scatter:
	with st.spinner("Loading explorer..."):
		html_to_show = load_scattertext(FPATHS['eda-scattertext'])
		components.html(html_to_show, width=1200, height=800, scrolling=True)
else:
    st.empty()