# Databricks notebook source
from geoscan import Geoscan

# COMMAND ----------

df = spark.sql("SELECT * FROM databricks_tourism_workspace.default.gowalla_checkins")

# COMMAND ----------

geoscan = Geoscan() \
    .setLatitudeCol("latitude") \
    .setLongitudeCol("longitude") \
    .setPredictionCol("cluster") \
    .setEpsilon(400) \
    .setMinPts(1)

model = geoscan.fit(df)

# COMMAND ----------

df = model.transform(df)

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gowalla_checkins")
