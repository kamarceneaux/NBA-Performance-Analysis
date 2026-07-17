import argparse
from typing import Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, LongType, BooleanType, DoubleType
)

# Define the schema for the foundation dataset
# Define all the tables

PROFILES_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("college", StringType(), True),
    StructField("draft_year", IntegerType(), True),
    StructField("draft_round", IntegerType(), True),
    StructField("draft_pick", IntegerType(), True),
    StructField("height", IntegerType(), True),
    StructField("weight", IntegerType(), True),
    StructField("country", StringType(), True),
    StructField("nba_debut_year", IntegerType(), True),
    StructField("all_nba_appearances", IntegerType(), True),
    StructField("is_drafted", BooleanType(), True),
])

STATS_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("season", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("team_name_abbr", StringType(), True),
    StructField("comp_name_abbr", StringType(), True),
    StructField("pos", StringType(), True),
    StructField("games", IntegerType(), True),
    StructField("games_started", IntegerType(), True),
    StructField("mp", IntegerType(), True),
    StructField("fg", IntegerType(), True),
    StructField("fga", IntegerType(), True),
    StructField("fg_pct", FloatType(), True),
    StructField("fg3", IntegerType(), True),
    StructField("fg3a", IntegerType(), True),
    StructField("fg3_pct", FloatType(), True),
    StructField("fg2", IntegerType(), True),
    StructField("fg2a", IntegerType(), True),
    StructField("fg2_pct", FloatType(), True),
    StructField("efg_pct", FloatType(), True),
    StructField("ft", IntegerType(), True),
    StructField("fta", IntegerType(), True),
    StructField("ft_pct", FloatType(), True),
    StructField("orb", IntegerType(), True),
    StructField("drb", IntegerType(), True),
    StructField("trb", IntegerType(), True),
    StructField("ast", IntegerType(), True),
    StructField("stl", IntegerType(), True),
    StructField("blk", IntegerType(), True),
    StructField("tov", IntegerType(), True),
    StructField("pf", IntegerType(), True),
    StructField("pts", IntegerType(), True),
    StructField("pts_per_game", DoubleType(), True),
    StructField("trb_per_game", DoubleType(), True),
    StructField("ast_per_game", DoubleType(), True),
    StructField("stl_per_game", DoubleType(), True),
    StructField("blk_per_game", DoubleType(), True),
    StructField("tov_per_game", DoubleType(), True),
    StructField("pts_per_min", DoubleType(), True),
    StructField("trb_per_min", DoubleType(), True),
    StructField("ast_per_min", DoubleType(), True),
    StructField("season_start_year", IntegerType(), True),

])

SALARIES_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("season", StringType(), True),
    StructField("teams", StringType(), True),
    StructField("salary", LongType(), True),
    StructField("is_two_way", BooleanType(), True),
    StructField("is_sub_minimum", BooleanType(), True),
    StructField("salary_pct_of_cap", DoubleType(), True),
    StructField("season_start_year", IntegerType(), True),
])

SALARY_CAP_SCHEMA = StructType([
    StructField("season", StringType(), False),
    StructField("salary_cap", IntegerType(), False),
    StructField("season_start_year", IntegerType(), True),
])

ADV_STATS_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("season", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("team_name_abbr", StringType(), True),
    StructField("comp_name_abbr", StringType(), True),
    StructField("pos", StringType(), True),
    StructField("games", IntegerType(), True),
    StructField("mp", IntegerType(), True),
    StructField("per", FloatType(), True),
    StructField("ts_pct", FloatType(), True),
    StructField("fg3a_per_fga_pct", FloatType(), True),
    StructField("fta_per_fga_pct", FloatType(), True),
    StructField("orb_pct", FloatType(), True),
    StructField("drb_pct", FloatType(), True),
    StructField("trb_pct", FloatType(), True),
    StructField("ast_pct", FloatType(), True),
    StructField("stl_pct", FloatType(), True),
    StructField("blk_pct", FloatType(), True),
    StructField("tov_pct", FloatType(), True),
    StructField("usg_pct", FloatType(), True),
    StructField("ows", FloatType(), True),
    StructField("dws", FloatType(), True),
    StructField("ws", FloatType(), True),
    StructField("ws_per_48", FloatType(), True),
    StructField("obpm", FloatType(), True),
    StructField("dbpm", FloatType(), True),
    StructField("bpm", FloatType(), True),
    StructField("vorp", FloatType(), True),
    StructField("season_start_year", IntegerType(), True)
])

STANDINGS_SCHEMA = StructType([
    StructField("season", StringType(), False),
    StructField("team_name_abbr", StringType(), True),
    StructField("overall", StringType(), True),
    StructField("wins", IntegerType(), True),       
    StructField("losses", IntegerType(), True),     
    StructField("win_pct", DoubleType(), True),     
    StructField("season_start_year", IntegerType(), True),
])

# Code that will help with connecting to the virtual machine
def parse_args():
    """Parse command-line arguments for input/output directories.

    This code was generated by AI to assist in setting up for our Virtual Machine we setup.
    """
    parser = argparse.ArgumentParser(
        description="NBA data cleaning pipeline (PySpark)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/raw/",
        help="Path (local or HDFS) to the directory containing the raw CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/cleaned/",
        help="Path (local or HDFS) where cleaned Parquet files will be written."
    )
    return parser.parse_args()


def create_spark_session():
    """
    Initialize a SparkSession tuned for the cleaning workload.

    This was created with the help of AI to ensure we have the right configurations for our data cleaning tasks.
    """
    return (
        SparkSession.builder
        .appName("CleanProcessing")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

# Function to load datasets with predefined schemas
def load_datasets(spark: SparkSession, input_dir: str):
    """
    Load the raw CSV datasets into Spark DataFrames with predefined schemas 
    """

    profiles_df = spark.read.parquet(f"{input_dir}/player_profiles")

    stats_df = spark.read.parquet(f"{input_dir}/nba_stats")

    adv_stats_df = spark.read.parquet(f"{input_dir}/nba_advanced_player_stats")

    standings_df = spark.read.parquet(f"{input_dir}/nba_expanded_standings_2000_2026")

    salaries_df = spark.read.parquet(f"{input_dir}/nba_salaries")
    salaries_df = salaries_df.withColumnRenamed("salary", "player_salary")

    salary_cap_df = spark.read.parquet(f"{input_dir}/nba_salary_cap")
    salary_cap_df = salary_cap_df.withColumnRenamed("salary_cap", "league_salary_cap")

    return profiles_df, stats_df, adv_stats_df, standings_df, salaries_df, salary_cap_df

# Function to join datasets on common keys
def join_datasets( profiles_df: DataFrame, stats_df: DataFrame, adv_stats_df: DataFrame, standings_df: DataFrame, salaries_df: DataFrame, salary_cap_df: DataFrame):
    """
    Join the profiles, stats, salaries, and salary_cap DataFrames 
    on common keys to create a unified dataset for analysis
    """
    adv_stats_df = adv_stats_df.drop("season_start_year")
    salaries_df = salaries_df.drop("season_start_year")
    salary_cap_df = salary_cap_df.drop("season_start_year")

    # Keep only advanced-only columns before joining to avoid duplicate names
    # from overlapping shared stats columns.
    adv_stats_df = adv_stats_df.select(
        "player_id", "season",
        "per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct",
        "orb_pct", "drb_pct", "trb_pct", "ast_pct", "stl_pct", "blk_pct",
        "tov_pct", "usg_pct", "ows", "dws", "ws", "ws_per_48",
        "obpm", "dbpm", "bpm", "vorp"
    )

    # Join basic stats + advanced stats
    stats_full_df = stats_df.join(
        adv_stats_df,
        on=["player_id", "season"],
        how="left"
    )

    # Join with player profiles
    profiles_stats_df = stats_full_df.join(
        profiles_df,
        on="player_id",
        how="left"
    )

    # Join with salaries
    profile_stats_salaries_df = profiles_stats_df.join(
        salaries_df,
        on=["player_id", "season"],
        how="left"
    )

    # Join with salary cap
    profile_stats_salaries_salary_cap_df = profile_stats_salaries_df.join(
        salary_cap_df.select("season", "league_salary_cap"),
        on="season",
        how="left"
    )

    # Join with standings
    standings_join_df = standings_df.select(
        "season_start_year", "team_name_abbr", "wins", "losses", "win_pct"
    )
    full_df = profile_stats_salaries_salary_cap_df.join(
        standings_join_df,
        on=["season_start_year", "team_name_abbr"],
        how="left",
    )

    return full_df

def main():
    args = parse_args()

    spark = create_spark_session()

    print("[LOAD] Loading cleaned parquet data...")
    profiles_df, stats_df, adv_stats_df, standings_df, salaries_df, salary_cap_df = load_datasets(
        spark, args.input_dir
    )

    print("[JOIN] Joining datasets...")
    final_df = join_datasets(
        profiles_df, stats_df, adv_stats_df,
        standings_df, salaries_df, salary_cap_df
    )

    print("[DEBUG] Showing sample:")
    final_df.show(5)

    print("[WRITE] Writing final dataset...")
    # final_df = final_df.toDF(*list[Any](dict.fromkeys(final_df.columns)))
    final_df.write.mode("overwrite").parquet(args.output_dir)

    print("[DONE] Final dataset written successfully.")

    spark.stop()


if __name__ == "__main__":
    main()