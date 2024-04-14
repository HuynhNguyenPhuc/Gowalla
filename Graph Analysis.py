# Databricks notebook source
# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

# MAGIC %pip install graphframes

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from operator import add
import numpy as np

from graphframes import GraphFrame

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC # Utils

# COMMAND ----------

def loadGraph():
    vertices = spark.sql("SELECT * FROM databricks_tourism_workspace.default.graph_vertices")
    edges = spark.sql("SELECT * FROM databricks_tourism_workspace.default.graph_edges")
    return vertices, edges

def saveGraph(vertices, edges):
    vertices.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_vertices")
    edges.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_edges")

# COMMAND ----------

# MAGIC %md
# MAGIC # Import dataset (Done)

# COMMAND ----------

df = spark.sql("SELECT * FROM databricks_tourism_workspace.default.gowalla_checkins")

# COMMAND ----------

window = Window.orderBy("cluster")

df = df.withColumn("cluster", F.dense_rank().over(window))
df = df.withColumn("cluster", F.col("cluster").cast("int"))

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("databricks_tourism_workspace.default.gowalla_checkins")

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Modeling (Done)

# COMMAND ----------

def getTourist(df):
    window_spec = Window.partitionBy("user").orderBy("checkinTime")

    # Add lag features
    df_sample = df.withColumn("prevTimestamp", F.lag(F.col("checkinTime")).over(window_spec))

    # Calculate time difference
    df_sample = df_sample.withColumn("timeDiff", F.unix_timestamp(F.col("checkinTime")) - F.unix_timestamp(F.col("prevTimestamp")))
    
    # Create a macro to identify each instance where the difference between two dates is larger than three days
    df_sample = df_sample.withColumn("newSegment", F.when(F.col("timeDiff") > (3 * 24 * 60 * 60), 1).otherwise(0))
    
    # Add a column to calculate the cumulative sum of the newSegment column
    df_sample = df_sample.withColumn("segment", F.sum("newSegment").over(window_spec))

    tourists = df_sample.groupBy("user", "segment").agg(F.F.collect_list(F.struct("cluster", "latitude", "longitude", "checkinTime")).alias("tourists"))
    tourists = tourists.groupBy("user").agg(F.F.collect_list("tourists").alias("tourists"))

    return tourists

# COMMAND ----------

# Get tourist dataset
tourists = getTourist(df)
tourists.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gowalla_tourist")

tourists = tourists.select("user", F.explode("tourists").alias("tourist"))
tourists = tourists.withColumn("index", F.monotonically_increasing_id())
tourists = tourists.select("user", "index", F.explode("tourist").alias("checkin"))

window_spec = Window.partitionBy("user", "index").orderBy("checkin")
tourists = tourists.withColumn("prevCheckin", F.lag(F.col("checkin")).over(window_spec))

# COMMAND ----------

edges = tourists.select(
    F.col("prevCheckin.cluster").alias("src"),
    F.col("checkin.cluster").alias("dst")
).where(F.col("src").isNotNull() & F.col("dst").isNotNull())

edges = edges.where(F.col("src") != F.col("dst"))
edges = edges.groupBy("src", "dst").agg(F.count("*").alias("normal"))

# COMMAND ----------

incoming_weights = edges.groupBy("dst").agg(F.sum("normal").alias("incoming_weight"))
outgoing_weights = edges.groupBy("src").agg(F.sum("normal").alias("outgoing_weight"))

vertices = incoming_weights.join(outgoing_weights, incoming_weights.dst == outgoing_weights.src, how="full_outer") \
    .select(F.coalesce(incoming_weights.dst, outgoing_weights.src).alias("id"),
            (F.coalesce(incoming_weights.incoming_weight, F.lit(0)) + F.coalesce(outgoing_weights.outgoing_weight, F.lit(0))).alias("normal"))

# COMMAND ----------

location_graph = GraphFrame(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pagerank (Done)

# COMMAND ----------

pagerank_results = location_graph.pageRank(resetProbability=0.15, tol = 1e-5)

pagerank_results.vertices.withColumnRenamed("weight", "pagerank").write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_vertices")
pagerank_results.edges.withColumnRenamed("weight", "pagerank").write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Number of points in a cluster (Done)

# COMMAND ----------

clusters = df.groupBy("cluster").agg(F.count("*").cast("int").alias("num_points"))

# COMMAND ----------

vertices, edges = loadGraph()

# COMMAND ----------

clusters.createOrReplaceTempView("clusters")
vertices.createOrReplaceTempView("vertices")

vertices = spark.sql("""
    SELECT id, c.num_points, normal, pagerank
    FROM vertices v JOIN clusters c ON v.id = c.cluster
    """)

# COMMAND ----------

saveGraph(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label Propagation Algorithm (Done)

# COMMAND ----------

vertices, edges = loadGraph()

# COMMAND ----------

g = GraphFrame(vertices, edges)

vertices = g.labelPropagation(maxIter=10)

windowSpec = Window.orderBy("label")
vertices = vertices.withColumn("label", F.dense_rank().over(windowSpec))

# COMMAND ----------

saveGraph(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Popular Tour Route (Done)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition Probabilty Matrix (Done)

# COMMAND ----------

total_weights = edges.groupBy(F.col("src").alias("new_src")).agg(F.sum(F.col("normal")).alias("total_weights"))

# COMMAND ----------

edges = edges.join(total_weights, edges.src == total_weights.new_src, "left"). \
        withColumn("transfer_prob", F.col("normal") / F.col("total_weights")). \
        select("src", "dst", "transfer_prob", "normal", "pagerank")

# COMMAND ----------

saveGraph(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Markov Clustering (Done)

# COMMAND ----------

vertices, edges = loadGraph()

# COMMAND ----------

vertices_test = vertices.filter(F.col("id") < 1000)
edges_test = edges.filter((F.col("src") < 1000) & (F.col("dst") < 1000))

# COMMAND ----------

def matrix_multiply(A, B):
    """
    Performs matrix multiplication of two CoordinateMatrices.
    
    Args:
    A (CoordinateMatrix): First matrix.
    B (CoordinateMatrix): Second matrix.
    
    Returns:
    CoordinateMatrix: Resultant matrix after multiplication.
    """
    try:
        A_rdd = A.entries.map(lambda x: (x.j,(x.i,x.value))) 
        B_rdd = B.entries.map(lambda x: (x.i,(x.j,x.value))) 
        interm_rdd = A_rdd.join(B_rdd).map(lambda x: ((x[1][0][0],x[1][1][0]),(x[1][0][1]*x[1][1][1])))
        C_rdd = interm_rdd.reduceByKey(add).map(lambda x: MatrixEntry(x[0][0],x[0][1],x[1])) 
        return CoordinateMatrix(C_rdd)
    except Exception as e:
        logging.error(f"An error occurred in matrix multiplication: {str(e)}")

def matrix_multiply_mod(a, b):
    """
    Performs matrix multiplication in BlockMatrix style.
    
    Args:
    a (CoordinateMatrix): First matrix.
    b (CoordinateMatrix): Second matrix.
    
    Returns:
    CoordinateMatrix: Resultant matrix after multiplication.
    """
    try:
        bmat_a = a.toBlockMatrix()
        b_tanspose= b.transpose()
        bmat_b_tanspose=b_tanspose.toBlockMatrix()
        bmat_result= bmat_a.multiply(bmat_b_tanspose)
        return bmat_result.toCoordinateMatrix()
    except Exception as e:
        logging.error(f"An error occurred in modified matrix multiplication: {str(e)}")

def normalize_mat(df):
    """
    Normalize the matrix by calculating L1 norm.
    
    Args:
    df (DataFrame): DataFrame representing the coordinate matrix.
    
    Returns:
    DataFrame: DataFrame of normalized matrix.
    """
    try:
        cols = df.columns
        df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt')
        tdf = df.groupby('dest').agg({'wt':'sum'}).withColumnRenamed('dest','dest_t').withColumnRenamed('sum(wt)','total_t')
        df = df.join(tdf,df.dest==tdf.dest_t)
        df = df.withColumn('new_wts', F.col('wt').cast('float')/F.col('total_t'))
        df = df.select('src','dest','new_wts')
        df = df.withColumnRenamed('src',cols[0]).withColumnRenamed('dest',cols[1]).withColumnRenamed('new_wts',cols[2])
        return df
    except Exception as e:
        logging.error(f"An error occurred in matrix normalization: {str(e)}")

def expand_mat(df,power,blockstyle=True):
    """
    Calculate the nth power of a matrix A.
    
    Args:
    df (DataFrame): DataFrame of the coordinate matrix A.
    power (int): Exponent to which the matrix should be raised.
    blockstyle (bool): Calculate matrix multiplication block style or by simple RDD joins.
    
    Returns:
    DataFrame: DataFrame of A^n matrix with source, destination, and weight columns.
    """
    try:
        cols = df.columns
        cdf =  CoordinateMatrix(df.rdd.map(tuple))
        rdf = cdf
        if blockstyle:
            for i in range(power-1):
                rdf = matrix_multiply_mod(rdf,cdf)
        else:
            for i in range(power-1):
                rdf = matrix_multiply(rdf,cdf)
        rdf_rdd = rdf.entries.map(lambda x: (x.i,x.j,x.value))
        result_df = rdf_rdd.toDF()
        result_df = result_df.withColumnRenamed('_1',cols[0]).withColumnRenamed('_2',cols[1]).withColumnRenamed('_3',cols[2])
        return result_df
    except Exception as e:
        logging.error(f"An error occurred in matrix expansion: {str(e)}")

def inflate_mat(df,inflate_size):
    """
    Raise each element of the matrix to the given power.
    
    Args:
    df (DataFrame): DataFrame of the coordinate matrix.
    inflate_size (int or float): Power to which each element should be raised.
    
    Returns:
    DataFrame: DataFrame of inflated matrix with source, destination, and weight columns.
    """
    try:
        cols = df.columns
        df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt')
        df = df.withColumn('new_wts', F.col('wt')**inflate_size)
        df = df.select('src','dest','new_wts')
        df = df.withColumnRenamed('src',cols[0]).withColumnRenamed('dest',cols[1]).withColumnRenamed('new_wts',cols[2])
        df = normalize_mat(df)
        return df
    except Exception as e:
        logging.error(f"An error occurred in matrix inflation: {str(e)}")

def prune_mat(df,threshold):
    """
    Prune the matrix if the weights are below a certain threshold.
    
    Args:
    df (DataFrame): DataFrame of the coordinate matrix.
    threshold (float): Threshold below which weights are ignored.
    
    Returns:
    DataFrame: Pruned DataFrame with source, destination, and weight columns.
    """
    try:
        cols = df.columns
        df = df.filter(F.col(cols[2])>threshold)
        return df
    except Exception as e:
        logging.error(f"An error occurred in matrix pruning: {str(e)}")

def converged(df1,df2):
    """
    Check for convergence by calculating the difference between the weights.
    
    Args:
    df1 (DataFrame): DataFrame of the coordinate matrix 1.
    df2 (DataFrame): DataFrame of the coordinate matrix 2.
    
    Returns:
    bool: True if matrices are converged, False otherwise.
    """
    try:
        cols1 = df1.columns
        cols2 = df2.columns
        df1 = df1.withColumnRenamed(cols1[0],'src1').withColumnRenamed(cols1[1],'dest1').withColumnRenamed(cols1[2],'wt1').persist()
        df2 = df2.withColumnRenamed(cols2[0],'src2').withColumnRenamed(cols2[1],'dest2').withColumnRenamed(cols2[2],'wt2').persist()
        df1.count()
        df2.count()

        @udf('int')
        def np_allclose(a,b):
            return int(np.allclose(a, b))

        df = df2.join(df1,(df1.src1==df2.src2) & (df1.dest1==df2.dest2), 'left').persist()
        df.count()
        df = df.fillna({'wt1':0})
        df = df.withColumn('allclose',np_allclose(F.col('wt1'),F.col('wt2'))).persist()

        if df.count() == df.filter(df.allclose==1).count():
            df.unpersist()
            return True
        else:
            df.unpersist()
            return False
    except Exception as e:
        logging.error(f"An error occurred in convergence check: {str(e)}")

def get_clusters(df):
    """
    Fetch clusters from the converged matrix.
    
    Args:
    df (DataFrame): DataFrame of the coordinate matrix.
    
    Returns:
    DataFrame: DataFrame of the clusters.
    """
    try:
        cols = df.columns
        df = df.withColumnRenamed(cols[0],'src').withColumnRenamed(cols[1],'dest').withColumnRenamed(cols[2],'wt')
        diagonals = df.filter((df.src==df.dest)&(df.wt>0)).select('src').distinct().collect()
        ids = [r[0] for r in diagonals]
        fdf = df.filter(df.src.isin(ids)).groupby('src').agg(F.collect_list('dest').alias("value"))
        fdf = fdf.rdd.zipWithIndex().toDF().withColumnRenamed('_1','nodes').withColumnRenamed('_2','cluster')
        fdf = fdf.select('cluster','nodes')
        return fdf
    except Exception as e:
        logging.error(f"An error occurred in cluster extraction: {str(e)}")

# COMMAND ----------

def runScaledMCL(matrix, expansion=2, inflation=2, loop_value=1, iterations=100, pruning_threshold=0.001, pruning_frequency=1, convergence_check_frequency=1):
    """
    Run the scaled Markov Clustering algorithm.

    Args:
    matrix (DataFrame): Input DataFrame.
    expansion (int): Expansion rate.
    inflation (int): Inflation rate.
    loop_value (int): Value for self-loops.
    iterations (int): Number of iterations.
    pruning_threshold (float): Pruning threshold.
    pruning_frequency (int): Pruning frequency.
    convergence_check_frequency (int): Convergence check frequency.

    Returns:
    DataFrame: Result of the MCL algorithm.
    """
    try:
        # Initialize variables
        result_matrix = None

        # Iterate through the specified number of iterations
        for i in range(iterations):
            # Perform MCL steps
            # Step 1: Expansion
            expanded_matrix = expand_mat(matrix, expansion)

            # Step 2: Inflation
            inflated_matrix = inflate_mat(expanded_matrix, inflation)

            # Step 3: Pruning
            pruned_matrix = prune_mat(inflated_matrix, pruning_threshold)

            # Check for convergence
            if i % convergence_check_frequency == 0:
                if converged(matrix, pruned_matrix):
                    logging.info(f"MCL algorithm converged after {i} iterations.")
                    break

            # Update the matrix for the next iteration
            matrix = pruned_matrix

        # Extract clusters from the converged matrix
        clusters_df = get_clusters(pruned_matrix)

        return clusters_df

    except Exception as e:
        logging.error(f"An error occurred in MCL algorithm: {str(e)}")


# COMMAND ----------

results = runScaledMCL(matrix = edges_test.select("src", "dst", "normal"), iterations = 5)

# COMMAND ----------

results.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gowalla_mcl")

# COMMAND ----------



# COMMAND ----------


