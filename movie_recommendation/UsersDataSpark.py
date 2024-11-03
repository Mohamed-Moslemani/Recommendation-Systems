from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *
import matplotlib.pyplot as plt 
import seaborn as sns 


spark = SparkSession.builder \
        .appName("RecommendationSystemMovies") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()  

file_path = 'data//Netflix_User_Ratings.csv'
df = spark.read.csv(file_path,header=True,inferSchema=True)

def nulls_counter(df):
    null_counts = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])
    null_counts.show()

def shapeChecker(df):
    nrows = df.count()
    ncols = len(df.columns)

    return [nrows,ncols]

def maxOfColumn(df,column):
    maxi = df.select(max(col(column))).collect()[0][0]
    return maxi 

def load_data(df):
    """
    Showcases the dataframe, and inspects its columns.

    Parameters: 
    -df: Spark=loaded user ratings dataframe - csv format read in spark.

    Returns: 
    -None
    """
    df.show(5)
    print("-"*50)
    df.printSchema()
    print("-"*50)
    nrows,ncols = shapeChecker(df)
    print("# of rows: {}\n# of columns: {}".format(nrows,ncols))
    print("-"*50)
    nulls_counter(df)
    print("-"*50) 
    max_rating = maxOfColumn(df,'Rating')
    print("Rating movies Maximum: {}".format(max_rating))
    print("-"*50)
    

def plot_ratings_distribution(df: DataFrame):
    """
    Plots the distribution of ratings in a Spark DataFrame.

    Parameters:
    - df: Spark DataFrame

    Returns:
    - None
    """
    ratings_dist = df.groupBy("Rating").count().orderBy("Rating")
    ratings_dist_pd = ratings_dist.toPandas()
    
    ratings_dist_pd.plot(kind='bar', x='Rating', y='count', legend=False)
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()


load_data(df)
plot_ratings_distribution(df=df)



