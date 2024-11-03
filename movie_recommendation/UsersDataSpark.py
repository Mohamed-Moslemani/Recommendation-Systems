from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *

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

def load_data(file_path):

    spark = SparkSession.builder \
        .appName("RecommendationSystemMovies") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()  
    
    df = spark.read.csv(file_path,header=True,inferSchema=True)
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
    spark.stop()
    

load_data('data//Netflix_User_Ratings.csv')