import pandas as pd
import numpy as np
import seaborn as sns
#from sklearn.cluster import cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.metrics as skmet
import errors as err
import scipy.optimize as opt

""" Tools to support clustering: correlation heatmap, normaliser and scale 
(cluster centres) back to original scale, check for mismatching entries """


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)


def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def get_diff_entries(df1, df2, column):
    """ Compares the values of column in df1 and the column with the same 
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match. """

    import pandas as pd  # to be sure

    # merge dataframes keeping all rows
    df_out = pd.merge(df1, df2, on=column, how="outer")
    print("total entries", len(df_out))
    # merge keeping only rows in common
    df_in = pd.merge(df1, df2, on=column, how="inner")
    print("entries in common", len(df_in))
    df_in["exists"] = "Y"

    # merge again
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")

    # extract columns without "Y" in exists
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()

    return diff_list

""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   

# Start of main program funtion to read Forest area (% of land area) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_forest = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", skiprows=4)
print(data_forest)


# Start of 2nd program funtion Average precipitation in depth (mm per year) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_precipitation = pd.read_csv("API_AG.LND.PRCP.MM_DS2_en_csv_v2_5456478.csv", skiprows=4)
print(data_precipitation)

# drop rows with nan's in 2020
#data_forest = data_forest[data_forest["1990"].notna()]
#data_forest = data_forest[data_forest["2000"].notna()]
data_forest = data_forest[data_forest["2010"].notna()]
data_forest = data_forest[data_forest["2020"].notna()]
print(data_forest)
print(data_forest.describe())

# drop rows with nan's in 2020
#data_precipitation = data_precipitation[data_precipitation["1990"].notna()]
#data_precipitation = data_precipitation[data_precipitation["2000"].notna()]
data_precipitation = data_precipitation[data_precipitation["2010"].notna()]
data_precipitation = data_precipitation[data_precipitation["2020"].notna()]
print(data_precipitation)
print(data_precipitation.describe())

'''
df_precipitation_2020 = data_precipitation[["Country Name", "2020"]].copy()
df_forest_2020 = data_forest[["Country Name", "2020"]].copy()

print(df_precipitation_2020)
print(df_forest_2020)

print(df_precipitation_2020.describe())
print(df_forest_2020.describe())

df_2020 = pd.merge(df_precipitation_2020, df_forest_2020, on="Country Name", how="outer")
# rename columns
df_2020 = df_2020.rename(columns={"2020_x":"2020_forest", "2020_y":"2020_precipitation"})
print(df_2020.describe())
df_2020.to_excel("pre_for2020.xlsx")


pd.plotting.scatter_matrix(df_2020, figsize=(12, 12), s=5, alpha=0.8)

#array([[<AxesSubplot:xlabel='precipitation', ylabel='precipitation'>,<AxesSubplot:xlabel='forest', ylabel='precipitation'>],[<AxesSubplot:xlabel='precipitation', ylabel='forest'>,<AxesSubplot:xlabel='forest', ylabel='forest'>]], dtype=object)

corr = df_2020.corr()
print(corr)

map_corr(df_2020)
plt.show()

df_ex = df_2020[["2020_forest", "2020_precipitation" ]] # extract the four columns for clustering
df_ex = df_ex.dropna() # entries with one nan are useless
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])

# reset_index() moved the old index into column index
# remove before clustering
df_ex = df_ex.drop("index", axis=1)

print(df_ex.iloc[0:15])

# normalise
df, df_min, df_max = scaler(df_ex)
print(df)
print(df_min)
print(df_max)

print()
print("n score")


# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_2020) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_ex, labels))
    
    


df_cluster = f_ex[["1990_precipitation", "1990_forest", "2000_precipitation", "2000_forest", "2010_precipitation", "2010_forest", "2020_precipitation", "2020_forest"]].copy()


'''