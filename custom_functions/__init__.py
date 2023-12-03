"""
Custom Functions From Throughout the Coding Dojo Data Science Program
__author__ = James Irving, Brenda Hungerford
"""
from .project_functions import *
from . import eda_functions as eda
# from . import _eda_functions_plotly as eda_plotly
from . import evaluate
from . import evaluate_admin as evaluate_save
from . import model_insights_functions as insights
from . import deployment_functions as deploy
from . import nlp_functions as nlp
from . import data_enrichment as data
from . import utils as utils
# from . import evaluate_app_functions as deploy


def show_code(function):
	"""Uses inspect modulem to retrieve source code for function.
	Displays as pthon-syntax Markdown code.
	
	Note: Python highlighting doesn't work correctly on VS Code or Google Colab
	"""
	import inspect
	from IPython.display import display, Markdown
	code = inspect.getsource(function)
	md = "```python" +'\n' + code + "\n" + '```' 
	display(Markdown(md))

