import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# Initialize argument parser stuff
parser = argparse.ArgumentParser()
parser.add_argument("--input-path",  required=True)
parser.add_argument("--output-path", required=True)
args = parser.parse_args()

spark = SparkSession.builder \
    .appName("CompositeValueScore") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# Read Data
raw = spark.read.parquet(args.input_path)

# Perform some quality checks to ensure we have the expected columns before proceeding
df = raw.filter(
    (F.col("player_salary").isNotNull()) &
    (F.col("player_salary") > 0) &
    (F.col("games") >= 20) &
    (F.col("mp") >= 200) &
    (F.col("team_name_abbr") != "2TM") &
    (F.col("team_name_abbr") != "3TM") &
    (F.col("team_name_abbr") != "4TM") &
    (F.col("team_name_abbr") != "5TM") &
    (F.col("team_name_abbr") != "6TM")
)

df = df.withColumn("player_salary", F.col("player_salary").cast(DoubleType()))
df = df.withColumn("salary_pct_of_cap", F.col("salary_pct_of_cap").cast(DoubleType()))

# Composite Value Score Calculation:
# Weighted per-game box-score contribution.
# Weights based on common APM regression coefficients, full definition in our timeline doc:
#   PTS × 1.0   — baseline scoring
#   TRB × 1.2   — possessions recovered / second chances
#   AST × 1.5   — playmaking creates higher-quality shots
#   STL × 2.0   — steals end opponent possessions and can create fast breaks
#   BLK × 2.0   — rim protection deters drives
#   TOV × −1.0  — lost possessions penalized
df = df.withColumn(
    "production_score",
    (
        1.0 * F.col("pts_per_game")
      + 1.2 * F.col("trb_per_game")
      + 1.5 * F.col("ast_per_game")
      + 2.0 * F.col("stl_per_game")
      + 2.0 * F.col("blk_per_game")
      - 1.0 * F.col("tov_per_game")
    )
)

df = df.withColumn(
    "salary_cap_share",
    F.greatest(F.col("salary_pct_of_cap"), F.lit(0.01))
)


# CVS = production_score / salary_cap_share
df = df.withColumn(
    "composite_value_score",
    F.col("production_score") / F.col("salary_cap_share")
)

# Availability
# Account for shortened seasons so players in lockout OR COVID years aren't unfairly penalized.
df = df.withColumn(
    "games_possible",
    F.when(F.col("season") == "2011-12", F.lit(66))
     .when(F.col("season") == "2019-20", F.lit(73))
     .when(F.col("season") == "2020-21", F.lit(72))
     .otherwise(F.lit(82))
)

df = df.withColumn(
    "availability_rate",
    F.least(F.col("games") / F.col("games_possible"), F.lit(1.0))
)


"""
AVAILABILITY-ADJUSTED CVS
Scale raw CVS by how often the player was on the court as a enhanced metric.
"""
df = df.withColumn(
    "availability_adj_cvs",
    F.col("composite_value_score") * F.col("availability_rate")
)


# Missed Game Cost
df = df.withColumn(
    "missed_game_cost",
    F.col("player_salary") * (1.0 - F.col("availability_rate"))
)


# OUTPUT SCRIPT
output = df.select(
    "player_id", "name", "season", "season_start_year",
    "team_name_abbr", "pos", "age",
    "height", "weight", "country", "college",
    "draft_year", "draft_round", "draft_pick", "is_drafted",
    "games", "games_started", "mp",
    "games_possible", "availability_rate",
    "pts_per_game", "trb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game", "tov_per_game",
    "per", "ts_pct", "ws", "ows", "dws", "ws_per_48",
    "bpm", "obpm", "dbpm", "vorp", "usg_pct",
    "player_salary", "salary_cap", "salary_pct_of_cap",
    "salary_cap_share",
    "is_two_way", "is_sub_minimum",
    "wins", "losses", "win_pct",
    "production_score",
    "composite_value_score",
    "availability_adj_cvs",
    "missed_game_cost"
)

output.coalesce(4).write.mode("overwrite").parquet(args.output_path)

row_count = output.count()
print("=" * 60)
print("COMPOSITE VALUE SCORE — COMPLETE")
print("=" * 60)

spark.stop()