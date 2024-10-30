from pyspark import SparkSession
import pandas as pd


def load_data(file_path):

    spark = SparkSession.builder \
        .appName("SparkTest") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    df = pd.read_csv(file_path)
    df.show(5)

    spark.stop()

    return df

def main():
    load_data("data//Netflix_User_Ratings.csv")
if __name__ == 'main':
    main()