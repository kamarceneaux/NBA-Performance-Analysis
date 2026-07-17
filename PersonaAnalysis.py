"""
Persona / archetype valuation: read joined parquet from HDFS, label player-seasons
with rule-based archetypes, write metrics + aggregates for the Streamlit dashboard.

  spark-submit PersonaAnalysis.py \\
      --input-path  hdfs:///.../final-data \\
      --output-path hdfs:///.../persona_analysis
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType

parser = argparse.ArgumentParser()
parser.add_argument("--input-path",  default="hdfs:///user/azureuser/final-data")
parser.add_argument("--output-path", default="hdfs:///user/azureuser/analytics/archetype_valuation")
args = parser.parse_args()

spark = (
    SparkSession.builder
    .appName("NBA_Archetype_Valuation")
    .config("spark.sql.shuffle.partitions", "50")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(">>> Reading joined dataset from:", args.input_path)
df = spark.read.parquet(args.input_path)

# 20+ GP, paid minutes, non-null WS (multi-team junk already dropped upstream)
qualified = df.filter(
    (F.col("games") >= 20) &
    (F.col("player_salary") > 0) &
    (F.col("mp") > 0) &
    (F.col("ws").isNotNull())
)

print(f">>> Qualified player-seasons: {qualified.count()}")

per36 = qualified.withColumn(
    "min_per_game", F.col("mp") / F.col("games")
).withColumn(
    "pts_per36",  (F.col("pts")  / F.col("mp")) * 36
).withColumn(
    "ast_per36",  (F.col("ast")  / F.col("mp")) * 36
).withColumn(
    "trb_per36",  (F.col("trb")  / F.col("mp")) * 36
).withColumn(
    "stl_per36",  (F.col("stl")  / F.col("mp")) * 36
).withColumn(
    "blk_per36",  (F.col("blk")  / F.col("mp")) * 36
).withColumn(
    "fg3a_per36", (F.col("fg3a") / F.col("mp")) * 36
).withColumn(
    "fg2a_per36", (F.col("fg2a") / F.col("mp")) * 36
).withColumn(
    "tov_per36",  (F.col("tov")  / F.col("mp")) * 36
).withColumn(
    "salary_millions", F.col("player_salary") / 1_000_000.0
).withColumn(
    "salary_share", F.col("salary_pct_of_cap")
)


def assign_archetype(
    pos, usg_pct, pts_per36, ts_pct,
    fg3a_per36, stl_per36, blk_per36, ast_per36,
    ast_pct, blk_pct, orb_pct, trb_per36,
    ws_per_48,     fg3a_per_fga
):
    if any(v is None for v in [pos, usg_pct, pts_per36, ts_pct,
                                fg3a_per36, stl_per36, blk_per36,
                                ast_per36, ast_pct, blk_pct, orb_pct,
                                trb_per36, ws_per_48, fg3a_per_fga]):
        return "Role Player / Glue Guy"

    # score each label, take max (ties pick earlier key in this dict)
    scores = {
        "Elite Scorer":          0,
        "3-and-D Wing":          0,
        "Playmaking Engine":     0,
        "Rim Protector":         0,
        "Stretch Big":           0,
        "Two-Way Anchor":        0,
        "Combo Guard":           0,
        "Role Player / Glue Guy": 0,
    }

    is_guard  = pos in ("PG", "SG", "G")
    is_wing   = pos in ("SF", "SG-SF", "SF-SG", "F")
    is_big    = pos in ("PF", "C", "C-PF", "PF-C", "F-C")

    if usg_pct >= 25:      scores["Elite Scorer"]      += 2
    if pts_per36 >= 20:    scores["Elite Scorer"]      += 2
    if ts_pct >= 0.56:     scores["Elite Scorer"]      += 1
    if usg_pct >= 22 and pts_per36 >= 18:
                           scores["Elite Scorer"]      += 1

    if fg3a_per_fga >= 0.45:  scores["3-and-D Wing"]  += 2
    if fg3a_per36 >= 5.0:     scores["3-and-D Wing"]  += 1
    if (stl_per36 + blk_per36) >= 2.5: scores["3-and-D Wing"] += 2
    if is_wing:               scores["3-and-D Wing"]  += 1
    if ast_per36 < 3.5:       scores["3-and-D Wing"]  += 1

    if ast_pct >= 28:      scores["Playmaking Engine"] += 3
    if ast_per36 >= 7.0:   scores["Playmaking Engine"] += 2
    if is_guard:           scores["Playmaking Engine"] += 1
    if usg_pct >= 22:      scores["Playmaking Engine"] += 1

    if blk_pct >= 4.0:     scores["Rim Protector"]    += 3
    if blk_per36 >= 2.0:   scores["Rim Protector"]    += 2
    if is_big:             scores["Rim Protector"]    += 1
    if orb_pct >= 8.0:     scores["Rim Protector"]    += 1
    if trb_per36 >= 9.0:   scores["Rim Protector"]    += 1

    if is_big and fg3a_per_fga >= 0.30: scores["Stretch Big"] += 3
    if is_big and fg3a_per36 >= 3.5:    scores["Stretch Big"] += 2
    if ast_per36 < 3.0:                 scores["Stretch Big"] += 1

    if (stl_per36 + blk_per36) >= 3.0: scores["Two-Way Anchor"] += 2
    if ws_per_48 >= 0.12:               scores["Two-Way Anchor"] += 2
    if blk_pct >= 2.5 and stl_per36 >= 1.0: scores["Two-Way Anchor"] += 2

    if is_guard and pts_per36 >= 14 and ast_per36 >= 3.5:
        scores["Combo Guard"] += 3
    if is_guard and fg3a_per_fga >= 0.30:
        scores["Combo Guard"] += 1
    if is_guard and usg_pct < 25:
        scores["Combo Guard"] += 1

    scores["Role Player / Glue Guy"] = 1

    return max(scores, key=scores.get)


archetype_udf = F.udf(assign_archetype, StringType())

print(">>> Assigning archetypes...")
labeled = per36.withColumn(
    "archetype",
    archetype_udf(
        F.col("pos"),
        F.col("usg_pct"),
        F.col("pts_per36"),
        F.col("ts_pct"),
        F.col("fg3a_per36"),
        F.col("stl_per36"),
        F.col("blk_per36"),
        F.col("ast_per36"),
        F.col("ast_pct"),
        F.col("blk_pct"),
        F.col("orb_pct"),
        F.col("trb_per36"),
        F.col("ws_per_48"),
        F.col("fg3a_per_fga_pct"),
    )
)

# value_score = vorp - λ*salary_share; λ=5 → ~1.75 VORP to break even at max-ish cap hit
LAMBDA = 5.0
SEASON_GAMES = 82.0

player_metrics = labeled.withColumn(
    "cost_per_ws",
    F.when(F.col("ws") > 0,
           F.col("player_salary") / F.col("ws")
    ).otherwise(None)
).withColumn(
    "value_score",
    F.col("vorp") - F.lit(LAMBDA) * F.col("salary_share")
).withColumn(
    "availability",
    (F.col("games") / F.lit(SEASON_GAMES)).cast(DoubleType())
).withColumn(
    "cost_per_available_ws",
    F.when(
        (F.col("ws") > 0) & (F.col("availability") > 0),
        F.col("player_salary") / (F.col("ws") * F.col("availability"))
    ).otherwise(None)
).withColumn(
    # rough output per $1M: pts, trb, ast, stl, blk, tov weighted
    "composite_index",
    (
        F.col("pts_per_game") +
        1.2 * F.col("trb_per_game") +
        1.5 * F.col("ast_per_game") +
        2.0 * F.col("stl_per_game") +
        2.0 * F.col("blk_per_game") -
        F.col("tov_per_game")
    ) / F.col("salary_millions")
)

print(">>> Computing archetype aggregations...")

archetype_season = player_metrics.groupBy(
    "season", "season_start_year", "archetype"
).agg(
    F.count("*").alias("player_count"),
    F.avg("cost_per_ws").alias("avg_cost_per_ws"),
    F.percentile_approx("cost_per_ws", 0.5).alias("median_cost_per_ws"),
    F.avg("value_score").alias("avg_value_score"),
    F.avg("composite_index").alias("avg_composite_index"),
    F.avg("ws").alias("avg_win_shares"),
    F.avg("vorp").alias("avg_vorp"),
    F.avg("per").alias("avg_per"),
    F.avg("salary_millions").alias("avg_salary_millions"),
    F.sum("player_salary").alias("total_archetype_salary"),
    F.avg("availability").alias("avg_availability"),
    F.avg("ts_pct").alias("avg_ts_pct"),
    F.avg("bpm").alias("avg_bpm"),
    F.avg("ws_per_48").alias("avg_ws_per_48"),
).withColumn(
    "cost_per_ws_millions",
    F.col("avg_cost_per_ws") / 1_000_000.0
).withColumn(
    "total_archetype_salary_millions",
    F.col("total_archetype_salary") / 1_000_000.0
)

player_archetype_output = player_metrics.select(
    "player_id",
    "name",
    "season",
    "season_start_year",
    "team_name_abbr",
    "pos",
    "archetype",
    "age",
    "games",
    "mp",
    "pts_per_game",
    "ast_per_game",
    "trb_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "pts_per36",
    "ast_per36",
    "trb_per36",
    "ts_pct",
    "usg_pct",
    "ws",
    "ws_per_48",
    "vorp",
    "bpm",
    "per",
    "player_salary",
    "salary_millions",
    "salary_pct_of_cap",
    "cost_per_ws",
    "cost_per_available_ws",
    "value_score",
    "composite_index",
    "availability",
    "wins",
    "losses",
    "win_pct",
    "all_nba_appearances",
    "height",
    "weight",
    "country",
    "is_drafted",
)

window_spec = Window.partitionBy("season", "archetype").orderBy(
    F.col("value_score").desc()
)

top_players_per_archetype = (
    player_archetype_output
    .withColumn("archetype_rank", F.rank().over(window_spec))
    .filter(F.col("archetype_rank") <= 5)
)

print(">>> Writing outputs...")

player_archetype_output.write.mode("overwrite") \
    .partitionBy("season_start_year") \
    .parquet(f"{args.output_path}/player_archetype_metrics")

archetype_season.write.mode("overwrite") \
    .parquet(f"{args.output_path}/archetype_season_summary")

top_players_per_archetype.write.mode("overwrite") \
    .parquet(f"{args.output_path}/archetype_top_players")

print(">>> Archetype valuation pipeline complete.")
print(f"    Output written to: {args.output_path}")

print("\n>>> Archetype distribution across all seasons:")
archetype_counts = player_archetype_output.groupBy("archetype") \
    .count().orderBy(F.col("count").desc())
archetype_counts.show(truncate=False)

print("\n>>> Sample: 2023-24 archetype cost-per-WS (millions):")
archetype_season.filter(F.col("season") == "2023-24") \
    .select("archetype", "player_count", "cost_per_ws_millions",
        "avg_value_score", "avg_win_shares") \
    .orderBy("cost_per_ws_millions") \
    .show(truncate=False)

spark.stop()