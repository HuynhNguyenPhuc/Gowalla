# Databricks notebook source
# MAGIC %pip install geoscan

# COMMAND ----------

import pyspark.sql.functions as F
from geoscan import Geoscan

# COMMAND ----------

df = spark.sql("SELECT * FROM databricks_tourism_workspace.default.gowalla_checkins")

df = df.select("user", "checkinTime", "latitude", "longitude")

# COMMAND ----------

geoscan = Geoscan() \
    .setLatitudeCol("latitude") \
    .setLongitudeCol("longitude") \
    .setPredictionCol("cluster") \
    .setEpsilon(400) \
    .setMinPts(2)

model = geoscan.fit(df)

# COMMAND ----------

df = model.transform(df)

# COMMAND ----------

## Remove null clusters (outliers)
df = df.filter(F.col("cluster").isNotNull())

## Calculate average of latitude and longtitude for points in one cluster
location = df.groupBy("cluster").agg(F.avg("latitude").alias("latitude"), F.avg("longitude").alias("longitude"))

# COMMAND ----------

df_temp = df.select("user", "checkinTime", "cluster")
df = df_temp.join(location, on=(df_temp.cluster == location.cluster)).select("user", "checkinTime", "latitude", "longitude", df_temp.cluster)

# COMMAND ----------

display(df)

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gowalla_checkins")
