import argparse
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import create_map, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, LongType, DoubleType, BooleanType
)

TEAM_MAP = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "New Jersey Nets": "NJN",
    "Charlotte Hornets": "CHO",
    "Charlotte Bobcats": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Vancouver Grizzlies": "VAN",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Hornets": "NOH",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Seattle SuperSonics": "SEA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

# Define the schema for the foundation dataset
# Define all the tables

PROFILES_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("college", StringType(), True),
    StructField("draft_year", FloatType(), True),
    StructField("draft_round", FloatType(), True),
    StructField("draft_pick", FloatType(), True),
    StructField("height", IntegerType(), True),
    StructField("weight", IntegerType(), True),
    StructField("country", StringType(), True),
    StructField("nba_debut_year", FloatType(), True),
    StructField("all_nba_appearances", IntegerType(), True),
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
])

SALARIES_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("season", StringType(), True),
    StructField("team", StringType(), True),
    StructField("salary", StringType(), True),
])

SALARY_CAP_SCHEMA = StructType([
    StructField("season", StringType(), False),
    StructField("salary_cap", IntegerType(), False),
])

ADV_STATS_SCHEMA = StructType([
    StructField("player_id", StringType(), False),
    StructField("season", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("team_name_abbr", StringType(), True),
    StructField("pos", StringType(), True),
    StructField("g", IntegerType(), True),
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
])

STANDINGS_SCHEMA = StructType([
    StructField("season", StringType(), True),
    StructField("team", StringType(), True),
    StructField("overall", StringType(), True),
])

# Code that will help with connecting to the virtual machine

def load_csv(spark, path, schema):
    """
    Read a CSV with an explicit schema. header=True skips the header row,
    and we disable type inference so everything follows our schema exactly.
    """
    return (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(path)
    )

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

    This was created w ith the help of AI to ensure we have the right configurations for our data cleaning tasks.
    """
    return (
        SparkSession.builder
        .appName("CleanProcessing")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

# Cleaning functions

def add_season_start_year(df: DataFrame, season_col: str = "season") -> DataFrame:
    """
    Extract the starting year from a season string like '2025-26' -> 2025.
     This will make it easier to join with the salary cap data later on if the calculations are right.
    """
    return df.withColumn(
        "season_start_year",
        F.substring(F.col(season_col), 1, 4).cast(IntegerType())
    )


def strip_carriage_returns(df: DataFrame) -> DataFrame:
    """
    Remove the style '\\r' characters from every StringType column that was causing issues.

    The code in this function was generated with AI-assistance,
    """
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(
                field.name,
                F.regexp_replace(F.col(field.name), r"\r", "")
            )
    return df

def clean_player_profiles(df: DataFrame) -> DataFrame:
    """
    Clean the player_profiles table:
      The goal of this function to ensure the player_profiles table is clean and consistent
       for later joins and analysis.

       written by Kameron Arceneaux
    """
    df = strip_carriage_returns(df)

    # safety check
    # shouldnt happen but if a player_id is missing then filter it out
    df = df.filter(F.col("player_id").isNotNull() & (F.col("player_id") != ""))

    # fill in missing values with defaults. If a -1.0 shows up in the draft columns, it is a sign a player was undrafted
    df = (
        df
        .fillna({"college": "None"})
        .fillna({
            "draft_year": -1.0,
            "draft_round": -1.0,
            "draft_pick": -1.0,
            "nba_debut_year": -1.0,
        })
    )

    # float --> int casting
    for col_name in ["draft_year", "draft_round", "draft_pick", "nba_debut_year"]:
        df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))

    # Making a column to see if a player was drafted. Should make analysis easier later on...
    df = df.withColumn(
        "is_drafted",
        F.when(F.col("draft_year") == -1, False).otherwise(True)
    )

    return df

def clean_nba_stats(df: DataFrame) -> DataFrame:
    """
    This is a function for some quality of life changes such as dropping all nulls,
    cleaning how to deal with multi-team players, and adding some per-game and per-minute stats for easier analysis later on.

     written by Kameron Arceneaux
    """
    df = strip_carriage_returns(df)
    df = df.filter(
        F.col("season").isNotNull()
        & (F.col("season") != "")
        & (F.col("season") != "N/A")
        & (F.col("season") != "0")
    )

    # Handle players who played for multiple teams in a season  in this section.
    multi_team_tags = {"2TM", "3TM", "4TM"}

    # Tag each row
    df = df.withColumn(
        "is_multi_team_agg",
        F.col("team_name_abbr").isin(multi_team_tags)
    )

    # For player-seasons that have an aggregate row, keep the aggregate.
    # For player-seasons without one, keep the single-team row as-is.
    agg_keys = (
        df
        .filter(F.col("is_multi_team_agg"))
        .select("player_id", "season")
        .distinct()
    )
    agg_rows = df.filter(F.col("is_multi_team_agg"))
    single_rows = df.filter(~F.col("is_multi_team_agg"))
    single_rows_no_dup = single_rows.join(
        agg_keys,
        on=["player_id", "season"],
        how="left_anti"
    )
    df = agg_rows.unionByName(single_rows_no_dup)
    df = df.drop("is_multi_team_agg")

    # validation of the shoots
    standard_pct_cols = ["fg_pct", "fg3_pct", "fg2_pct", "ft_pct"]
    for col_name in standard_pct_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < 0.0, None)
            .when(F.col(col_name) > 1.0, None)
            .otherwise(F.col(col_name))
        )

    # EFG% = (FG + 0.5 * 3P) / FGA  →  can legitimately exceed 1.0.
    # https://www.breakthroughbasketball.com/stats/effective-field-goal-percentage
    df = df.withColumn(
        "efg_pct",
        F.when(F.col("efg_pct") < 0.0, None)
        .when(F.col("efg_pct") > 1.5, None)
        .otherwise(F.col("efg_pct"))
    )

    # Typiucal per game analysis stats
    df = (
        df
        .withColumn("pts_per_game", F.round(F.col("pts") / F.col("games"), 2))
        .withColumn("ast_per_game", F.round(F.col("ast") / F.col("games"), 2))
        .withColumn("trb_per_game", F.round(F.col("trb") / F.col("games"), 2))
        .withColumn("stl_per_game", F.round(F.col("stl") / F.col("games"), 2))
        .withColumn("blk_per_game", F.round(F.col("blk") / F.col("games"), 2))
        .withColumn("tov_per_game", F.round(F.col("tov") / F.col("games"), 2))
    )

    # Some basic stats per minute analysis which could be useful later on...
    df = (
        df
        .withColumn("pts_per_min",
                    F.when(F.col("mp") > 0,
                           F.round(F.col("pts") / F.col("mp"), 4))
                    .otherwise(None))
        .withColumn("ast_per_min",
                    F.when(F.col("mp") > 0,
                           F.round(F.col("ast") / F.col("mp"), 4))
                    .otherwise(None))
        .withColumn("trb_per_min",
                    F.when(F.col("mp") > 0,
                           F.round(F.col("trb") / F.col("mp"), 4))
                    .otherwise(None))
    )

    df = add_season_start_year(df)

    return df


def clean_nba_salaries(df: DataFrame) -> DataFrame:
    """
    Goals...
    Parse the messy salary column dealing 2W contracts and sub-minimum contracts.
    Flag two-way contracts in a boolean column ``is_two_way``.
    Drop rows where cleaned salary is null/
    If a player was traded mid-season and appears on multiple teams,
        sum the salary, concatenate team names, keep two-way flag
        if ANY of the rows were two-way.
      - Add season_start_year.\

    written by Marcus Reese
    """
    df = strip_carriage_returns(df)

    # Drop rows with no season
    df = df.filter(
        F.col("season").isNotNull()
        & (F.col("season") != "")
    )

    # Flag two-way contracts
    df = df.withColumn(
        "is_two_way",
        F.col("salary").contains("(TW)")
    )

    # Flag sub-minimum contracts
    df = df.withColumn(
        "is_sub_minimum",
        F.col("salary").contains("< Minimum")
    )

    #Clean up columsn with textual contents and only extract the numbers
    # When writing the regex AI assisted.
    df = df.withColumn(
        "salary_cleaned",
        F.regexp_replace(F.col("salary"), r"\(TW\)", "")  # remove (TW)
    )
    df = df.withColumn(
        "salary_cleaned",
        F.regexp_replace(F.col("salary_cleaned"), r"< Minimum", "")  # remove < Minimum
    )
    df = df.withColumn(
        "salary_cleaned",
        F.trim(F.col("salary_cleaned"))
    )
    df = df.withColumn(
        "salary_cleaned",
        F.when(
            F.col("salary_cleaned") == "", None
        ).otherwise(
            F.col("salary_cleaned").cast(LongType())
        )
    )
    null_salary_count = df.filter(F.col("salary_cleaned").isNull()).count()
    print(f"Dropping {null_salary_count} salary rows with "
          f"missing salary amounts.")

    # Drop rows where we have no usable salary figure
    df = df.filter(F.col("salary_cleaned").isNotNull())
    df = df.drop("salary").withColumnRenamed("salary_cleaned", "salary")

    # A player traded mid-season can appear on two teams with partial salaries.
    #     # We sum salary across teams so each player has one row per season.
    df = (
        df.groupBy("player_id", "season")
        .agg(
            F.sum("salary").alias("salary"),
            F.concat_ws(" / ", F.collect_set("team")).alias("teams"),
            F.max(F.col("is_two_way").cast(IntegerType())).alias("is_two_way_int"),
            F.max(F.col("is_sub_minimum").cast(IntegerType())).alias("is_sub_min_int"),
        )
    )

    #int ---> boolean
    df = (
        df
        .withColumn("is_two_way",
                    F.col("is_two_way_int").cast(BooleanType()))
        .withColumn("is_sub_minimum",
                    F.col("is_sub_min_int").cast(BooleanType()))
        .drop("is_two_way_int", "is_sub_min_int")
    )

    df = add_season_start_year(df)

    return df


def clean_salary_cap(df: DataFrame) -> DataFrame:
    """
    Clean the salary_cap table:
      - Strip \\r.
      - Add season_start_year for joins.
    """
    df = strip_carriage_returns(df)
    df = add_season_start_year(df)
    return df

def clean_advanced_stats(df: DataFrame) -> DataFrame:
    """
    Clean advanced player stats:
    Remove bad rows, validate percentage columns, and handle multi-team players, add season_start_year
    Same logic as clean_nba_stats
    """

    df = strip_carriage_returns(df)

    # Remove invalid seasons
    df = df.filter(
        F.col("season").isNotNull()
        & (F.col("season") != "")
        & (F.col("season") != "N/A")
        & (F.col("season") != "0")
    )

    # Handle multi-team players
    multi_team_tags = {"2TM", "3TM", "4TM"}

    df = df.withColumn(
        "is_multi_team_agg",
        F.col("team_name_abbr").isin(multi_team_tags)
    )

    agg_keys = (
        df
        .filter(F.col("is_multi_team_agg"))
        .select("player_id", "season")
        .distinct()
    )

    agg_rows = df.filter(F.col("is_multi_team_agg"))
    single_rows = df.filter(~F.col("is_multi_team_agg"))

    single_rows_no_dup = single_rows.join(
        agg_keys,
        on=["player_id", "season"],
        how="left_anti"
    )

    df = agg_rows.unionByName(single_rows_no_dup)
    df = df.drop("is_multi_team_agg")

    # Validate advanced columns with correct scales:
    # - true ratios in [0, 1]
    # - Basketball-Reference percentage stats are percentage points (0..100+)
    ratio_cols = ["ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct"]
    pct_point_cols = [
        "orb_pct", "drb_pct", "trb_pct",
        "ast_pct", "stl_pct", "blk_pct",
        "tov_pct", "usg_pct",
    ]

    for col_name in ratio_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < 0.0, None)
            .when(F.col(col_name) > 1.0, None)
            .otherwise(F.col(col_name))
        )

    for col_name in pct_point_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < 0.0, None)
            .when(F.col(col_name) > 100.0, None)
            .otherwise(F.col(col_name))
        )

    # Basic sanity checks for advanced metrics
    df = df.withColumn(
        "ws",
        F.when(F.col("ws") < -50, None)
        .when(F.col("ws") > 50, None)
        .otherwise(F.col("ws"))
    )

    df = df.withColumn(
        "bpm",
        F.when(F.col("bpm") < -50, None)
        .when(F.col("bpm") > 50, None)
        .otherwise(F.col("bpm"))
    )

    df = add_season_start_year(df)

    return df

def clean_expanded_standings(df: DataFrame) -> DataFrame:
    """
    Clean expanded standings:
    Remove bad rows, parse wins/losses from 'overall', compute win percentage, add season_start_year
    """

    df = strip_carriage_returns(df)

    # Remove invalid rows
    df = df.filter(
        F.col("season").isNotNull()
        & (F.col("season") != "")
        & F.col("team").isNotNull()
        & (F.col("team") != "")
    )

    # Split 'overall' into wins/losses
    df = df.withColumn(
        "wins",
        F.split(F.col("overall"), "-")[0].cast(IntegerType())
    ).withColumn(
        "losses",
        F.split(F.col("overall"), "-")[1].cast(IntegerType())
    )

    # Compute win %
    df = df.withColumn(
        "win_pct",
        F.when(
            (F.col("wins") + F.col("losses")) > 0,
            F.round(F.col("wins") / (F.col("wins") + F.col("losses")), 4)
        ).otherwise(None)
    )

    # Add the team abbr to the DataFrame
    mapping_expr = create_map([lit(x) for x in sum(TEAM_MAP.items(), ())])

    df = df.withColumn(
        "team_name_abbr",
        mapping_expr[F.col("team")]
    )

    df = add_season_start_year(df)

    return df


def normalize_salary_to_cap(
        salaries_df: DataFrame,
        cap_df: DataFrame
) -> DataFrame:
    """
    Join cleaned salaries with the salary cap table and compute
    ``salary_pct_of_cap`` = salary / salary_cap. This can make it easier to compare player salaries across different eras with varying cap levels.

    Written by: Hammaad Alam
    """
    salaries_df = salaries_df.join(
        cap_df.select("season", "salary_cap"),
        on="season",
        how="left"
    )

    salaries_df = salaries_df.withColumn(
        "salary_pct_of_cap",
        F.when(
            F.col("salary_cap").isNotNull() & (F.col("salary_cap") > 0),
            F.round(F.col("salary") / F.col("salary_cap"), 6)
        ).otherwise(None)
    )

    return salaries_df


#####
def print_cleaning_summary(
        raw_counts: dict,
        clean_counts: dict,
        label: str = ""
):
    """Print a before/after row-count summary for the cleaning report."""
    print(f"\n{'=' * 60}")
    print(f"  CLEANING SUMMARY  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Table':<25} {'Raw Rows':>10} {'Clean Rows':>12} {'Dropped':>10}")
    print(f"  {'-' * 57}")
    for table in raw_counts:
        raw = raw_counts[table]
        clean = clean_counts.get(table, 0)
        print(f"  {table:<25} {raw:>10,} {clean:>12,} {raw - clean:>10,}")
    print(f"{'=' * 60}\n")


def validate_no_nulls_in_keys(df: DataFrame, key_cols: list, table_name: str):
    """Assert that key columns have no nulls after cleaning."""
    for col_name in key_cols:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            print(f"  [WARN] {table_name}.{col_name} has {null_count} "
                  f"null values after cleaning!")
        else:
            print(f"  [ OK ] {table_name}.{col_name} — no nulls.")


def validate_no_duplicates(df: DataFrame, key_cols: list, table_name: str):
    """Assert that key columns form a unique composite key."""
    total = df.count()
    distinct = df.select(key_cols).distinct().count()
    if total != distinct:
        print(f"  [WARN] {table_name} has {total - distinct} duplicate "
              f"rows on {key_cols}!")
    else:
        print(f"  [ OK ] {table_name} — unique on {key_cols} "
              f"({total:,} rows).")

def main():
    args = parse_args()
    spark = create_spark_session()

    input_dir = args.input_dir.rstrip("/")
    output_dir = args.output_dir.rstrip("/")

    # Load raw data
    print("[LOAD] Reading raw CSVs …")
    raw_profiles = load_csv(spark, f"{input_dir}/player_profiles.csv", PROFILES_SCHEMA)
    raw_stats = load_csv(spark, f"{input_dir}/nba_stats.csv", STATS_SCHEMA)
    raw_salaries = load_csv(spark, f"{input_dir}/nba_salaries.csv", SALARIES_SCHEMA)
    raw_salary_cap = load_csv(spark, f"{input_dir}/nba_salary_cap.csv", SALARY_CAP_SCHEMA)
    raw_adv_stats = load_csv(spark, f"{input_dir}/nba_advanced_player_stats.csv", ADV_STATS_SCHEMA)
    raw_standings = load_csv(spark, f"{input_dir}/nba_expanded_standings_2000_2026.csv", STANDINGS_SCHEMA)

    raw_counts = {
        "player_profiles": raw_profiles.count(),
        "nba_stats": raw_stats.count(),
        "nba_salaries": raw_salaries.count(),
        "nba_salary_cap": raw_salary_cap.count(),
        "nba_advanced_player_stats": raw_adv_stats.count(),
        "nba_expanded_standings_2000_2026": raw_standings.count(),
    }
    print(f"Logs for my reference. Raw row counts: {raw_counts}")

    # Clean each table
    print("\nCleaning player_profiles …")
    clean_profiles = clean_player_profiles(raw_profiles)
    clean_profiles.cache()

    print("Cleaning nba_stats …")
    clean_stats = clean_nba_stats(raw_stats)
    clean_stats.cache()

    print(" Cleaning nba_salaries …")
    clean_salaries = clean_nba_salaries(raw_salaries)
    clean_salaries.cache()

    print("Cleaning nba_salary_cap …")
    clean_cap = clean_salary_cap(raw_salary_cap)
    clean_cap.cache()

    print("Cleaning nba_advanced_player_stats …")
    clean_adv_stats = clean_advanced_stats(raw_adv_stats)
    clean_adv_stats.cache()

    print("Cleaning nba_expanded_standings_2000_2026 …")
    clean_standings = clean_expanded_standings(raw_standings)
    clean_standings.cache()

    print("Normalizing salaries to cap percentage …")
    clean_salaries = normalize_salary_to_cap(clean_salaries, clean_cap)
    clean_salaries.cache()

    clean_counts = {
        "player_profiles": clean_profiles.count(),
        "nba_stats": clean_stats.count(),
        "nba_salaries": clean_salaries.count(),
        "nba_salary_cap": clean_cap.count(),
        "nba_advanced_player_stats": clean_adv_stats.count(),
        "nba_expanded_standings_2000_2026": clean_standings.count(),
    }

    print_cleaning_summary(raw_counts, clean_counts)

    print("Key integrity checks:")
    validate_no_nulls_in_keys(clean_profiles, ["player_id"], "player_profiles")
    validate_no_nulls_in_keys(clean_stats, ["player_id", "season"], "nba_stats")
    validate_no_nulls_in_keys(clean_salaries, ["player_id", "season"], "nba_salaries")
    validate_no_nulls_in_keys(clean_cap, ["season"], "nba_salary_cap")
    validate_no_nulls_in_keys(clean_adv_stats, ["player_id", "season"], "nba_advanced_player_stats")
    validate_no_nulls_in_keys(clean_standings, ["season"], "nba_expanded_standings_2000_2026")

    print("\n clean_profiles:")
    clean_profiles.show(5, truncate=False)

    print("clean_stats:")
    clean_stats.select(
        "player_id", "season", "team_name_abbr", "pos",
        "games", "mp", "pts", "pts_per_game", "pts_per_min",
        "season_start_year"
    ).show(5, truncate=False)

    print("clean_salaries:")
    clean_salaries.show(5, truncate=False)

    # CSV ---> Parquet to enagegw ith.
    print(f"[WRITE] Writing cleaned Parquet to {output_dir} …")

    clean_profiles.write.mode("overwrite").parquet(
        f"{output_dir}/player_profiles"
    )
    clean_stats.write.mode("overwrite").parquet(
        f"{output_dir}/nba_stats"
    )
    clean_salaries.write.mode("overwrite").parquet(
        f"{output_dir}/nba_salaries"
    )
    clean_cap.write.mode("overwrite").parquet(
        f"{output_dir}/nba_salary_cap"
    )
    clean_adv_stats.write.mode("overwrite").parquet(
        f"{output_dir}/nba_advanced_player_stats"
    )
    clean_standings.write.mode("overwrite").parquet(
        f"{output_dir}/nba_expanded_standings_2000_2026"
    )

    print("[DONE] All six tables cleaned and written successfully.")
    spark.stop()


if __name__ == "__main__":
    main()