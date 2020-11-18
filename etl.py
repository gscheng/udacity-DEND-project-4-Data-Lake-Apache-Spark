import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import IntegerType, TimestampType, DateType, StructType, StructField, StringType, LongType, DoubleType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('CREDENTIALS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('CREDENTIALS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    '''
    Creates spark session
    '''
    
    print('Initializing Spark session')
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    print('Spark session created.')
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Processes the song data from S3 and creates the songs and artists tables
    '''
    
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    # Define song data schema
    song_data_schema = StructType([
        StructField('artist_id', StringType(), True),
        StructField('artist_latitude', DoubleType(), True),
        StructField('artist_longitude', DoubleType(), True),
        StructField('artist_location', StringType(), True),
        StructField('artist_name', StringType(), True),
        StructField('duration', DoubleType(), True),
        StructField('num_songs', IntegerType(), True),
        StructField('song_id', StringType(), True),
        StructField('title', StringType(), True),
        StructField('year', IntegerType(), True)])
    
    # read song data file
    print('reading in song data...')
    df = spark.read.json(song_data, schema=song_data_schema)

    # extract columns to create songs table
    df.createOrReplaceTempView('song_data')
    
    songs_table = spark.sql("""
                            SELECT DISTINCT
                                song_id,
                                LTRIM(RTRIM(title)) AS title,
                                artist_id,
                                IF(year=0,null,year) AS year,
                                duration
                            FROM song_data
                            """)
    
    
    # write songs table to parquet files partitioned by year and artist
    print('Writing songs table parquet file...')
    songs_table.write.partitionBy('year', 'artist_id').parquet(output_data + 'songs')

    # extract columns to create artists table
    artists_table = spark.sql("""
                              SELECT 
                                  artist_id, 
                                  artist_name, 
                                  IF(artist_location='' OR artist_location='None',null,artist_location) AS artist_location, 
                                  artist_latitude, 
                                  artist_longitude
                              FROM song_data
                              """)
    
    # write artists table to parquet files
    print('Writing artists table parquet file...')
    artists_table.write.parquet(output_data + 'artists')
    
    return songs_table, artists_table, df


def process_log_data(spark, input_data, output_data, song_df): 
    '''
    Processes log data file from S3, creates users and time table, and then create songplays table by joining columns in the log data and song data
    '''
    
    # Define log data schema
    log_data_schema = StructType([StructField("artist", StringType(), True),
                             StructField("auth", StringType(), True),
                             StructField("firstName", StringType(), True),
                             StructField("gender", StringType(), True),
                             StructField("itemInSession", LongType(), True),
                             StructField("lastName", StringType(), True),
                             StructField("length", DoubleType(), True),
                             StructField("level", StringType(), True),
                             StructField("location", StringType(), True),
                             StructField("method", StringType(), True),
                             StructField("page", StringType(), True),
                             StructField("registration", DoubleType(), True),
                             StructField("sessionId", LongType(), True),
                             StructField("song", StringType(), True),
                             StructField("status", LongType(), True),
                             StructField("ts", LongType(), True),
                             StructField("userAgent", StringType(), True),
                             StructField("userId", StringType(), True)])
    
    # get filepath to log data file
    log_data =  input_data + 'log_data/*/*/*.json'

    # read log data file
    print('Reading in log data...')
    df = spark.read.json(log_data, schema=log_data_schema)
    
    # filter by actions for song plays
    df = df.filter(df.page =='NextSong')
    
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create temp view for spark sql
    df.createOrReplaceTempView('log_data')
        
    # extract columns for users table    
    users_table = spark.sql("""
                                SELECT 
                                    user_id,
                                    first_name,
                                    last_name,
                                    gender,
                                    level
                                FROM (
                                        SELECT 
                                            timestamp AS start_time,
                                            userId AS user_id,
                                            firstname AS first_name,
                                            lastname AS last_name,
                                            gender,
                                            level,
                                            RANK() OVER (PARTITION BY userid ORDER BY timestamp DESC) AS rank
                                        FROM log_data 
                                    ) AS sub_query
                                WHERE rank = 1
                                ORDER BY user_id
                            """)
    
    # write users table to parquet files
    print('Writing users table parquet file...')
    users_table.write.parquet(output_data + 'users')

    # extract columns to create time table
    time_table = spark.sql("""
                           SELECT DISTINCT
                                timestamp AS start_time,
                                CAST(date_format(timestamp, "HH") AS INTEGER) AS hour,
                                CAST(date_format(timestamp, "dd") AS INTEGER) AS day,
                                CAST(weekofyear(timestamp) AS INTEGER) AS week,
                                CAST(month(timestamp) AS INTEGER) AS month,
                                CAST(year(timestamp) AS INTEGER) AS year,
                                CAST(dayofweek(timestamp) AS INTEGER) AS day_of_week
                           FROM log_data
                           """)
    
    
    # write time table to parquet files partitioned by year and month
    print('Writing time table parquet file...')
    time_table.write.partitionBy('year','month').parquet(output_data + 'time')
    
    # create temp view for spark sql
    song_df.createOrReplaceTempView('song_data')

    # extract columns from joined song and log datasets to create songplays table 
    print('Creating songplays table...')
    songplays_table = spark.sql("""
                                SELECT DISTINCT
                                    l.timestamp AS start_time,
                                    l.userId AS user_id,
                                    l.level,
                                    s.song_id,
                                    s.artist_id,
                                    l.sessionId AS session_id,
                                    l.location,
                                    l.useragent AS user_agent
                                FROM log_data AS l
                                LEFT JOIN song_data AS s ON ((l.song = s.title) AND (l.artist = s.artist_name) AND (l.length = s.duration))
                                """).withColumn('songplay_id', monotonically_increasing_id())

    
    songplays_table = songplays_table.select(['songplay_id','start_time','user_id','level','song_id', 'artist_id', 'session_id', 'location','user_agent']) 

    # write songplays table to parquet files partitioned by year and month
    print('Writing songplays table parquet file...')
    songplays_table.withColumn('year', year('start_time')).withColumn('month',month('start_time')).write.partitionBy('year','month').parquet(output_data + 'songplays')
    
    return users_table, time_table, songplays_table

def row_counts(spark, songs_table, artists_table, users_table, time_table, songplays_table):
    '''
    Gets row counts of each of the 5 tables created (songs, artists, users, time, songplays)
    '''
    
    print('Running sample queries...')
    print('Songs table row count: ', songs_table.count())
    print('Artists table row count: ', artists_table.count())
    print('Users table row count: ', users_table.count())
    print('Time table row count: ', time_table.count())
    print('Songplays table row count: ', songplays_table.count())
    
def show_top_rows(spark, songs_table, artists_table, users_table, time_table, songplays_table):
    '''
    Shows top 5 rows of each of the 5 tables created (songs, artists, users, time, songplays)
    '''
    
    print('Retrieving top 5 rows for each table...')
    print('\nSongs Table')
    songs_table.show(5,truncate=False)
    print('Artists Table')
    artists_table.show(5,truncate=False)
    print('Users Table')
    users_table.show(5,truncate=False)
    print('Time Table')
    time_table.show(5,truncate=False)
    print('Songplays Table')
    songplays_table.show(5)
    

def songplays_query(spark, songs_table, artists_table, users_table, time_table, songplays_table):
    '''
    Sample query using all tables created with Spark using SQL and joining them
    '''
    
    # Create temp tables for spark sql
    songs_table.createOrReplaceTempView('songs_table')
    artists_table.createOrReplaceTempView('artists_table')
    users_table.createOrReplaceTempView('users_table')
    time_table.createOrReplaceTempView('time_table')
    songplays_table.createOrReplaceTempView('songplays_table')
    
    result = spark.sql("""
                        SELECT DISTINCT
                            sp.songplay_id,
                            u.user_id,
                            s.song_id,
                            u.first_name,
                            u.last_name,
                            u.gender,
                            u.level,
                            sp.start_time,
                            a.artist_name,
                            s.title,
                            s.duration,
                            sp.location
                        FROM songplays_table AS sp
                        JOIN users_table AS u ON (u.user_id = sp.user_id)
                        JOIN artists_table AS a ON (a.artist_id = sp.artist_id)
                        JOIN songs_table AS s ON (s.song_id = sp.song_id)
                        JOIN time_table AS t ON (t.start_time = sp.start_time)
                        ORDER BY sp.start_time
                        LIMIT 100
                        """)
 
    print("\nJoined table schema:")
    result.printSchema()
    print("Loading user songplays...")
    result.show(truncate=False)
    

def main():
    '''
    Main Program
    '''
    
    start_time = datetime.now()
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-project-spark/"
   
    songs_table, artists_table, song_df = process_song_data(spark, input_data, output_data)    
    users_table, time_table, songplays_table = process_log_data(spark, input_data, output_data, song_df)

    print('ETL pipeline completed in: {}'.format(datetime.now() - start_time))
    etl_time = datetime.now()
    
    # Run sample queries
    print(row_counts(spark, songs_table, artists_table, users_table, time_table, songplays_table))
    print(show_top_rows(spark, songs_table, artists_table, users_table, time_table, songplays_table))
    songplays_query(spark, songs_table, artists_table, users_table, time_table, songplays_table)

    print('Sample queries completed in: {}'.format(datetime.now() - etl_time))
    print('Total run time: {}'.format(datetime.now() - start_time))


if __name__ == "__main__":
    main()
