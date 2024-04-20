# Databricks notebook source
import math
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from operator import add
import numpy as np

# COMMAND ----------

def loadGraph():
    vertices = spark.sql("SELECT * FROM databricks_tourism_workspace.default.graph_vertices")
    edges = spark.sql("SELECT * FROM databricks_tourism_workspace.default.graph_edges")
    return vertices, edges

def saveGraph(vertices, edges):
    vertices.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_vertices")
    edges.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("graph_edges")

# COMMAND ----------

vertices, edges = loadGraph()

# COMMAND ----------

# MAGIC %md
# MAGIC # Markov Clustering

# COMMAND ----------

vertices_test = vertices.filter(col("id") < 10000)
edges_test = edges.filter((col("src") < 10000) & (col("dst") < 10000))

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
        df = df.withColumn('new_wts', col('wt').cast('float')/col('total_t'))
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
        df = df.withColumn('new_wts', col('wt')**inflate_size)
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
        df = df.filter(col(cols[2])>threshold)
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
        df = df.withColumn('allclose',np_allclose(col('wt1'),col('wt2'))).persist()

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
        fdf = df.filter(df.src.isin(ids)).groupby('src').agg(collect_list('dest')).withColumnRenamed('collect_list(dest)','clusters')
        fdf = fdf.rdd.zipWithIndex().toDF().withColumnRenamed('_1','nodes_in_cluster').withColumnRenamed('_2','cluster_id')
        fdf = fdf.select('cluster_id','nodes_in_cluster')
        return fdf
    except Exception as e:
        logging.error(f"An error occurred in cluster extraction: {str(e)}")

# COMMAND ----------

def run_scaled_mcl(matrix, expansion=2, inflation=2, loop_value=1, iterations=100, pruning_threshold=0.001, pruning_frequency=1, convergence_check_frequency=1):
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

results = run_scaled_mcl(matrix = edges_test.select("src", "dst", "normal"), iterations= 1)

# COMMAND ----------

display(results)

# COMMAND ----------

results.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gowalla_mcl")

# COMMAND ----------

# MAGIC %md
# MAGIC # Application

# COMMAND ----------


