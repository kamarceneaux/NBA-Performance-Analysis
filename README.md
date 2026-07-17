# NBA Salary vs Performance Efficiency Analysis

## Project Overview
This project features a distributed data pipeline and interactive analytics dashboard designed to evaluate NBA player performance relative to their contract value. Rather than evaluating players solely on raw counting stats, this project measures "value per dollar" to identify market inefficiencies, best-value contracts, and optimal player archetypes. 

The pipeline extracts, cleans, and joins multi-season data, processing it in a distributed cluster environment to support three core analytical models.

## Architecture & Technologies
*   **Data Acquisition:** Custom Python web scrapers built with `BeautifulSoup` to extract datasets (performance, advanced stats, salaries, standings, and profiles) from Basketball Reference while adhering to server request limits.
*   **Distributed Storage:** Hadoop Distributed File System (HDFS) deployed across a two-node Microsoft Azure virtual network (Standard_D2as_v5 VMs) to ensure data consistency and reliability.
*   **Data Processing:** Apache Spark (PySpark) handles all data cleansing, normalization (e.g., era-adjusted salary caps), and complex table joins in a distributed manner.
*   **Frontend Dashboard:** Streamlit application providing interactive filtering, visualization, and player lookup capabilities based on the aggregated data.

## Key Analytical Models
### 1. Composite Value Score (CVS)
A position-agnostic metric that collapses traditional statistics into a weighted sum, penalized by turnovers, and divided by the player's percentage of the team salary cap. The metric is calculated as:

$$CVS = \frac{PTS + 1.2 \times TRB + 1.5 \times AST + 2 \times STL + 2 \times BLK - TOV}{\frac{Salary}{Team\ Salary\ Cap}}$$

*Note: The dashboard also introduces an Availability-Adjusted CVS to penalize missed games and track "wasted" salary cap space.*

### 2. Player Archetype Clustering
Utilizes a rule-based model to categorize players into 8 modern positional archetypes (e.g., *3-and-D Wing*, *Playmaking Engine*, *Two-Way Anchor*). Players are evaluated using a PySpark User Defined Function (UDF) across 14 statistical features, allowing front offices to determine the median market cost per Win Share for specific roles.

### 3. Salary Bucketing Analysis
Groups players into seven distinct salary tiers based on their cap percentage (e.g., 0-3%, 24-30%) using a PySpark `Bucketizer`. Players are then evaluated against their direct peers within the same financial bracket using Box Plus/Minus (BPM), True Shooting Percentage (TS%), and Win Shares per 48 (WS/48). 

## Repository Structure
*   `/scraping/`: Python scripts (`scrape_stats.py`, `scrape_profiles.py`, etc.) used to extract the raw CSV data from the web.
*   `/scripts/`: PySpark pipeline scripts for the ETL process.
    *   `01_clean_foundation_updated.py`: Normalizes raw data, handles multi-team mid-season trades, and converts to Parquet.
    *   `02_load_join_updated.py`: Joins the 6 clean tables into a unified analytical dataset.
    *   `03_analysis_cluster.py`, `PersonaAnalysis.py`, `composite_value_scores.py`: Execute the metric calculations and aggregations.
*   `/dashboard/`: Streamlit application files, featuring `main_app.py` and dedicated pages for each analytical view.

## Getting Started
To run this application in a distributed environment, you must have access to the cluster's master node and the required `.pem` key. 

**1. Initialize the Cluster**
Access the master node and start the distributed file system and resource manager:
```bash
ssh -i ~/Downloads/master_key.pem azureuser@<master-ip>
start-dfs.sh
start-yarn.sh
```

**2. Execute the PySpark Pipeline**
Submit the ETL and analysis jobs to the Spark cluster:
```bash
spark-submit ~/scripts/01_clean_foundation_updated.py \
  --input-dir hdfs:///user/azureuser/raw-data \
  --output-dir hdfs:///user/azureuser/cleaned-data

spark-submit ~/scripts/02_load_join_updated.py \
  --input-dir hdfs:///user/azureuser/cleaned-data \
  --output-dir hdfs:///user/azureuser/final-data

# Run analysis models
spark-submit ~/scripts/composite_value_score.py --input-dir hdfs:///user/azureuser/final-data --output-dir hdfs:///user/azureuser/analysis/composite_value_score
spark-submit ~/scripts/PersonaAnalysis.py --input-dir hdfs:///user/azureuser/final-data --output-dir hdfs:///user/azureuser/analysis/persona_analysis
spark-submit ~/scripts/03_analysis_cluster.py --input-dir hdfs:///user/azureuser/final-data --output-dir hdfs:///user/azureuser/analysis/bucket_values_score
```

**3. Launch the Dashboard**
Activate the Streamlit environment and launch the web application on port 8501:
```bash
source ~/streamlit-env/bin/activate
streamlit run /dashboard/main_app.py --server.port 8501 --server.address 0.0.0.0
```

## Contributors
*   **Kameron Arceneaux** - Data cleaning pipelines and CVS metric engineering.
*   **Hammaad Alam** - Data joining logic and Bucketing analysis.
*   **Marcus Reese** - Azure VM infrastructure configuration and web scraping. 
*(Project built for CSC 4740 – Big Data Technology)*
