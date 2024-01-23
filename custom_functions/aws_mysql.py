import pymysql
pymysql.install_as_MySQLdb()

from sqlalchemy import create_engine
from sqlalchemy_utils import create_database, database_exists
from urllib.parse import quote_plus
import json

def connect_to_aws_rds(creds_file = '/Users/codingdojo/.secret/aws-personal.json', include_engine=False):
    ## loading mysql credentials
    import pymysql
    pymysql.install_as_MySQLdb()
    with open(creds_file) as f:
        login = json.load(f)
# login.keys()

    ## create a new movies database
    # connect_str = f"mysql+pymysql://{login['user']}:{login['password']}@localhost/movies"
    host = login['host']
    port = login['port']
    password = quote_plus(login['password'])
    username = login['username']
    db_name = login['database']
    connect_str = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
    # connect_str

    ## Create the engine
    engine = create_engine(connect_str)
    conn = engine.connect()

    print(f'[i] Successfully connected to AWS RDS database: {db_name}')

    if include_engine:
        return conn, engine
    else:
        return conn
    