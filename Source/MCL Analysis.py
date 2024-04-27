# Databricks notebook source
# MAGIC %pip install networkx
# MAGIC %pip install --upgrade scipy

# COMMAND ----------

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pyspark.sql.functions as F

# COMMAND ----------

mcl = spark.sql("SELECT * FROM databricks_tourism_workspace.default.gowalla_mcl")

# COMMAND ----------

display(mcl)

# COMMAND ----------

graph_edges = spark.sql("SELECT * FROM databricks_tourism_workspace.default.graph_edges").select("src", "dst", F.col("normal").alias("weight"))

# COMMAND ----------

mcl_graph = mcl.select("nodes_in_cluster.src", F.explode("nodes_in_cluster.clusters").alias("dst"), "cluster_id")

# COMMAND ----------

mcl_graph = mcl_graph.select("src", "dst")

# COMMAND ----------

mcl_graph = mcl_graph.join(graph_edges, on=((mcl_graph.src == graph_edges.src) & (mcl_graph.dst == graph_edges.dst))).select(mcl_graph.src, mcl_graph.dst, graph_edges.weight)

# COMMAND ----------

mcl_graph.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("mcl_graph")

# COMMAND ----------

display(mcl_graph)

# COMMAND ----------

display(mcl_graph.orderBy("weight"))

# COMMAND ----------

mcl_df = mcl_graph.toPandas()

G = nx.from_pandas_edgelist(mcl_df, 'src', 'dst', ['weight'])

weights = [data['weight'] for _, _, data in G.edges(data=True)]

plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=False, node_color='skyblue', node_size=50, edge_color='black', width= 5*weights, font_size=15)
plt.title("Graph Visualization")
plt.show()
