"""
Salary-bucket value score: bucket players by salary_pct_of_cap, compare 
bpm/ws_per_48/ts_pct/usg_pct to cluster averages, scale diffs, and ouput a value_score
"""

import argparse

from pyspark.ml.feature import Bucketizer, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Assign Salary Groups
def assign_salary_groups(df):
    """
    Split salary_pct_of_cap into predefined salary_group buckets.
    """
    splits = [-1.0, 0.03, 0.07, 0.12, 0.18, 0.24, 0.30, float("inf")]

    bucketizer = Bucketizer(
        splits=splits,
        inputCol="salary_pct_of_cap",
        outputCol="salary_group",
        handleInvalid="skip",
    )

    return bucketizer.transform(df)

def compute_group_averages(df):
    """
    Compute group averages for weighted value-score inputs.
    """
    group_avg = df.groupBy("salary_group").agg(
        F.avg("bpm").alias("avg_bpm"),
        F.avg("ws_per_48").alias("avg_ws_per_48"),
        F.avg("ts_pct").alias("avg_ts_pct"),
        F.avg("usg_pct").alias("avg_usg_pct"),
    )
    return df.join(group_avg, on="salary_group")


def compute_differences(df):
    """
    Compute differences from salary-group averages for each weighted feature.
    """
    return (
        df.withColumn("bpm_diff", F.col("bpm") - F.col("avg_bpm"))
        .withColumn("ws_per_48_diff", F.col("ws_per_48") - F.col("avg_ws_per_48"))
        .withColumn("ts_pct_diff", F.col("ts_pct") - F.col("avg_ts_pct"))
        .withColumn("usg_pct_diff", F.col("usg_pct") - F.col("avg_usg_pct"))
    )

def scale_differences(df):
    """
    Scale weighted-feature differences to comparable units.
    """
    assembler = VectorAssembler(
        inputCols=["bpm_diff", "ws_per_48_diff", "ts_pct_diff", "usg_pct_diff"],
        outputCol="diff_raw",
        handleInvalid="skip",
    )
    df = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="diff_raw",
        outputCol="diff_scaled",
        withMean=True,
        withStd=True,
    )
    return scaler.fit(df).transform(df)


def compute_value_score(df):
    """
    Weighted value score using scaled salary-group-relative features:
    value_score = 0.40*bpm + 0.25*ws_per_48 + 0.20*ts_pct + 0.15*usg_pct
    """

    df = df.withColumn("diff_scaled_arr", vector_to_array(F.col("diff_scaled")))
    return df.withColumn(
        "value_score",
        0.40 * F.col("diff_scaled_arr").getItem(0)
        + 0.25 * F.col("diff_scaled_arr").getItem(1)
        + 0.20 * F.col("diff_scaled_arr").getItem(2)
        + 0.15 * F.col("diff_scaled_arr").getItem(3),
    )


def run_value_pipeline(df):
    """
    Run the value pipeline to compute the value score.
    """

    df = df.filter(
        F.col("salary_pct_of_cap").isNotNull()
        & F.col("bpm").isNotNull()
        & F.col("ws_per_48").isNotNull()
        & F.col("ts_pct").isNotNull()
        & F.col("usg_pct").isNotNull()
    )
    df = assign_salary_groups(df)
    df = compute_group_averages(df)
    df = compute_differences(df)
    df = scale_differences(df)
    df = compute_value_score(df)
    return df


def main():
    """
    Main function to run the value pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Salary-bucket value score (bpm/ws_per_48/ts_pct/usg_pct vs cluster)"
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("SalaryBucketValueScore").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = spark.read.parquet(args.input_path)

    # Same filters as composite_value_scores.py for comparable rows
    df = raw.filter(
        (F.col("player_salary").isNotNull())
        & (F.col("player_salary") > 0)
        & (F.col("games") >= 20)
        & (F.col("mp") >= 200)
        & (F.col("team_name_abbr") != "2TM")
        & (F.col("team_name_abbr") != "3TM")
        & (F.col("team_name_abbr") != "4TM")
        & (F.col("team_name_abbr") != "5TM")
        & (F.col("team_name_abbr") != "6TM")
    )

    df = df.withColumn("player_salary", F.col("player_salary").cast(DoubleType()))
    df = df.withColumn("salary_pct_of_cap", F.col("salary_pct_of_cap").cast(DoubleType()))

    scored = run_value_pipeline(df)

    # Core identifiers
    output_cols = [
        "player_id",
        "name",
        "season",
        "season_start_year",
        "team_name_abbr",
        "pos",
        "age",
        "games",
        "mp",
        "bpm",
        "ws_per_48",
        "ts_pct",
        "usg_pct",
        "salary_pct_of_cap",
        "salary_group",
        "avg_bpm",
        "avg_ws_per_48",
        "avg_ts_pct",
        "avg_usg_pct",
        "bpm_diff",
        "ws_per_48_diff",
        "ts_pct_diff",
        "usg_pct_diff",
        "value_score",
    ]
    present = [c for c in output_cols if c in scored.columns]
    output = scored.select(*present)

    output.coalesce(4).write.mode("overwrite").parquet(args.output_path)

    row_count = output.count()
    print("=" * 60)
    print("SALARY BUCKET VALUE SCORE — COMPLETE")
    print(f"  Rows written : {row_count}")
    print(f"  Output path  : {args.output_path}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()