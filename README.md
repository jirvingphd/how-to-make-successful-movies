# How to Make a Successful Movie

- James M. Irving

<img src="Images/movies-generic-header.png" width=300px>

## Business Problem

I have been hired to produce analyze IMDB's extensive publicly-available dataset, supplement it with financial data from TMDB's API, convert the raw data into a MySQL database, and then use that database for extracting insights and recommendations on how to make a successful movie.

I will use a combination of machine-learning-model-based insights and hypothesis testing to extract insights for our stakeholder.
    

  
### Specifications/Constraints    
- The stakeholder wants to focus on attributes of the movies themselves, vs the actors and directors connected to those movies. 
- They only want to include information related to movies released in the United States.
- They also did not want to include movies released before the year 2000.
- The stakeholder is particularly interested in how the MPAA rating, genre(s), runtime, budget, and production companies influence movie revenue and user-ratings.

        
# Methods

### `Part 1- Initial IMDB Data Processing.ipynb`

#### IMDB Movie Metadata
- I will download fresh movie metadata from IMDB's public datasets and filter out movies that meet the stakeholder's requirements/constraints.

- IMDB Provides Several Files with varied information for Movies, TV Shows, Made for TV Movies, etc.
    - Overview/Data Dictionary: <a href="https://www.imdb.com/interfaces/" target="_blank">https://www.imdb.com/interfaces/</a>

    - Downloads page: <a href="https://datasets.imdbws.com/" target="_blank">https://datasets.imdbws.com/</a>
- Files to use:
    - title.basics.tsv.gz
    - title.ratings.tsv.gz
    - title.akas.tsv.gz
  

###  `Part 2 - Extracting TMDB Data.ipynb`

#### Supplement Data from The Movie Database  (TMDB)'s

- I will extract MPAA rating and financial data for the movies using TMDB's API.

<img src="./Images/tmdb_logo_blue_long.svg" width=400px>
        
 
>"This product uses the TMDB API but is not endorsed or certified by TMDB." 
       
        
        
        
       

### `Part 3 - MySQL Database Construction`

- I will then normalize all IMDB movie data into a proper MySQL database.
    - MVP Version (included): Local Server Installation with Publicly-Available .sql file for recreationl.
    - AAB Version (future work): AWS-hosted RDS MySQL database. 
    

### `Part 4 - Hypothesis Testing`


- I will then use the MySQL database to answer several hypotheses about movie success.

### `Part 5 - Regression Model-Based Insights`

- Finally I will use Linear Regression and other machine learning models to predict movie revenue / ROI to extract insights and recommendations on what features of a movie are positive/negative predictors of success.
