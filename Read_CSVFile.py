# -*- coding: utf-8 -*-
"""
Created on Sun May 14 12:41:21 2023

@author: bhagy
"""
import pandas as pd

# Start of 1st program funtion Forest area (% of land area) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_forest = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", header=2)
print(data_forest)


# drop rows with nan's in 1990 and 2020
data_forest = data_forest[data_forest["1990"].notna()]
data_forest = data_forest[data_forest["2020"].notna()]

print(data_forest)

df_forest_2020 = data_forest[["Country Name", "1990", "2020"]].copy()

print(df_forest_2020)
print(df_forest_2020.describe())
