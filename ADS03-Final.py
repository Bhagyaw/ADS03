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

# Define a linear model
def linear_model(x, m, c):
    return m * x + c


def poly(x, a, b, c, d, e):
    """
    Calulates polynominal
    """
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 + e*x**4

    return f


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

# Start of main program

# funtion to read Forest area (% of land area) from 1990 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_forest = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", skiprows=4)
print(data_forest)


# funtion Average precipitation in depth (mm per year) from 1990 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_precipitation = pd.read_csv("API_AG.LND.PRCP.MM_DS2_en_csv_v2_5456478.csv", skiprows=4)
print(data_precipitation)

#selecting required columns for analysis
columns = ['Country Name', "Country Code", 'Indicator Name', '1990', '2020']

df_forest = data_forest[columns]
df_precipitation = data_precipitation[columns]

print(df_forest)
print(df_precipitation)

# Print summary statistics
print(df_forest.describe())
print(df_precipitation.describe())

# drop rows with nan's in 1990 and 2020
df_forest = df_forest[data_forest["1990"].notna()]
#data_forest = data_forest[data_forest["2000"].notna()]
#data_forest = data_forest[data_forest["2010"].notna()]
df_forest = df_forest[data_forest["2020"].notna()]
print(df_forest)


# drop rows with nan's in 1990 and 2020
df_precipitation = df_precipitation[data_precipitation["1990"].notna()]
#data_precipitation = data_precipitation[data_precipitation["2000"].notna()]
#data_precipitation = data_precipitation[data_precipitation["2010"].notna()]
df_precipitation = df_precipitation[data_precipitation["2020"].notna()]
print(df_precipitation)

# Print summary statistics
print(df_forest.describe())
print(df_precipitation.describe())



# Create new DataFrames containing only the relevant columns for '2020'
df_forest2020 = df_forest[["Country Name", "2020"]].copy()
df_precipitation2020 = df_precipitation[["Country Name", "2020"]].copy()

# Print summary statistics
print(df_forest2020.describe())
print(df_precipitation2020.describe())

# Merge the 'df_forest2020' and 'df_precipitation2020' DataFrames based on
# 'Country Name'. The 'how="outer"' argument specifies that entries
# not found in both DataFrames should be included
df_2020 = pd.merge(df_precipitation2020, df_forest2020, on="Country Name", how="outer")
print(df_2020)

#rename columns in df_2020
df_2020 = df_2020.rename(columns={"2020_x":"2020_forest", "2020_y":"2020_precipitation"})

# Print summary statistics for the 'df_2020' DataFrame
print(df_2020.describe())

# Save the 'df_2020' DataFrame to an Excel file
df_2020.to_excel("prefor2020.xlsx")

# Create a scatter matrix plot of the 'df_2020' DataFrame
scatter_matrix = pd.plotting.scatter_matrix(df_2020, figsize=(12, 12), s=5, alpha=0.8)

# Add titles legends and labels
plt.suptitle('Scatter Matrix Plot of df_2020', fontsize=18)
for ax in scatter_matrix.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    
# Add legends
for ax in scatter_matrix[:,0]:
    ax.yaxis.label.set_rotation(90)
    ax.yaxis.label.set_ha('right')
    ax.legend()

for ax in scatter_matrix[-1,:]:
    ax.xaxis.label.set_rotation(0)
    ax.xaxis.label.set_ha('right')
    ax.legend() 
    
plt.tight_layout()
plt.show()

 

#Finding the correlation using corr funtion. 
corr = df_2020.corr()
print(corr)

# Create a correlation plot of the 'df_2020' DataFrame
mapcorr = map_corr(df_2020)

# Add titles legends and labels
plt.suptitle('Correlation Plot of df_2020', fontsize=18)
plt.show()

#Heatmap using seaborn
sns.heatmap(corr, cmap="coolwarm", annot=True, annot_kws={"size": 10}, xticklabels=corr.columns, yticklabels=corr.columns)
plt.title("Correlation Plot Forest vs Precipitation for year 2020")
plt.tight_layout()
plt.show()


#Required to nomalise the data using scalar funtion.
#before clustering data,extract the two columns for clustering
df_ex = df_2020[["2020_forest", "2020_precipitation" ]] 
# entries with one nan are useless
df_ex = df_ex.dropna()
#Reset the indexing
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])

# reset_index() moved the old index into column index
# remove before clustering
df_ex = df_ex.drop("index", axis=1)
print(df_ex.iloc[0:15])

# normalise
df_cluster, df_min, df_max = scaler(df_ex)
print(df_cluster)
print(df_min)
print(df_max)

print('\n')



# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster) 
    # fit label on x,y data point pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    silhoutte_score = skmet.silhouette_score(df_cluster, labels)
    #print the calculated values.
    print("n score")
    print(ncluster)
    print(silhoutte_score)
    print('\n')

# set the number of clusters to 5 (chosen based on the silhouette scores)
ncluster = 5

# set up the clusterer with the number of expected clusters
kmeans = KMeans(n_clusters=ncluster)

# fit the data to the clusterer, storing the results in the kmeans object
kmeans.fit(df_cluster)

# get the labels assigned to each data point by the clusterer
labels = kmeans.labels_

# get the estimated cluster centers
cen = kmeans.cluster_centers_

# get the x-coordinates of the cluster centers
xcen = cen[:, 0]

# get the y-coordinates of the cluster centers
ycen = cen[:, 1]


# Define the colormap and color labels
cmap = plt.cm.get_cmap('tab10')
color_labels = range(len(labels))

# Create a scatter plot of the clustered data
plt.figure(figsize=(10.0, 5.0))
scatter = plt.scatter(df_cluster["2020_precipitation"], df_cluster["2020_forest"], s=40,
                      c=labels, cmap=cmap, alpha=0.8, edgecolors='none')

#In order to show  cluster membership and cluster centres, Create a scatter plot of the cluster centroids
plt.scatter(xcen, ycen, s=45, marker="d", color='k')

# Add a colorbar legend
cbar = plt.colorbar(scatter, ticks=color_labels)
cbar.ax.set_yticklabels(color_labels)

# Add a title and axis labels
plt.title("Clustered Data Forest area vs. Precipitation ", fontsize=16)
plt.xlabel("Forest area", fontsize=12)
plt.ylabel("Precipitation", fontsize=12)

plt.show()




# Define the variables to be fitted
x_data = df_2020["2020_forest"]
y_data = df_2020["2020_precipitation"]

#selecting required columns for analysis
input_forest = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", skiprows=4)
columns_fit = ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']
df_forest_fit = input_forest[columns_fit]
print(df_forest_fit)

# Create new DataFrames containing only the relevant columns for '2020'
df_forest_5Years = df_forest_fit[["Country Name", "1990", "1995", "2000", '2005', '2010', '2015', "2020"]].copy()
#rename columns in df_2020
df_forest_5Years = df_forest_5Years.rename(columns={"Country Name":"Years"})
# transpose the dataframe
df_forest_t = df_forest_5Years.transpose()
print(df_forest_t)

#Selecting required countries to analysis
df__forest_data = df_forest_t.iloc[: , [29, 35, 40, 81, 251]].copy()
print(df__forest_data)

# Create a DataFrame from df_forest_dset dataset
df_forest_data = pd.DataFrame({'Years': [1990, 1995, 2000, 2005, 2010, 2015, 2020],
                   'Brazil': [70.458021, 68.19619, 65.934359, 63.57092, 61.207482, 60.286715, 59.417478],
                   'Canada': [38.845512, 38.819247, 38.792982, 38.766226, 38.739471, 38.716438, 38.695513],
                   'China': [16.673325, 17.726847, 18.780497, 20.033048, 21.285597, 22.313094, 23.340596],
                   'United Kingdom': [11.48266, 11.846402, 12.210143, 12.427148, 12.644153, 13.040962, 13.185632],
                   'United States': [33.022308, 33.081594, 33.130174, 33.413084, 33.749407, 33.899723, 33.866926]
                   })
#df_forest_data = df_forest_data.set_index('Years', inplace=True)
print(df_forest_data)

# Create a line plot of the Forest area (% of land area) over time
plt.figure(figsize=(10.0, 6.0))
plt.plot(df_forest_data['Years'], df_forest_data['Brazil'], marker="o")
plt.plot(df_forest_data['Years'], df_forest_data['Canada'], marker="o")
plt.plot(df_forest_data['Years'], df_forest_data['China'], marker="o")
plt.plot(df_forest_data['Years'], df_forest_data['United Kingdom'], marker="o")
plt.plot(df_forest_data['Years'], df_forest_data['United States'], marker="o")

#labelling x axis and y axis
plt.xlabel('Year')
plt.ylabel('Forest area (% of land area)')

#Add legend to explain the each lineplot 
plt.legend(['Brazil', 'Canada', 'China', 'United Kingdom', 'United States'], loc='center left', bbox_to_anchor=(1, 0.5))
#Add Title to the graph
plt.title("Forest area (% of land area)")
# save the plot output as png
plt.savefig("Forest area.png")
#Display the plot
plt.show()


# defining exponential funtion
def exponential(t, n0, g):
    """
    Calculates exponential function with scale factor n0 and growth rate g.
    """

    t = t - 1990
    f = n0 * np.exp(g*t)

    return f

#Calculate the deforestation rate with exponential funtion
#Calculation for the Brazil
param1, covar1 = opt.curve_fit(
    exponential, df_forest_data['Years'], df_forest_data['Brazil'],
    p0=(1.2e12, 0.03))

print("Brazil Forest area (% of land area)", param1[0]/1e9)
print("Growth rate", param1[1])

#Calculation for the Canada
param2, covar2 = opt.curve_fit(
    exponential, df_forest_data['Years'], df_forest_data['Canada'],
    p0=(1.2e12, 0.03))

print("Canada Forest area (% of land area)", param2[0]/1e9)
print("Growth rate", param2[1])

#Calculation for the China
param3, covar3 = opt.curve_fit(
    exponential, df_forest_data['Years'], df_forest_data['China'],
    p0=(1.2e12, 0.03))

print("China Forest area (% of land area)", param3[0]/1e9)
print("Growth rate", param3[1])

#Calculation for the UK
param4, covar4 = opt.curve_fit(
    exponential, df_forest_data['Years'], df_forest_data['United Kingdom'],
    p0=(1.2e12, 0.03))

print("UK Forest area (% of land area)", param4[0]/1e9)
print("Growth rate", param4[1])

#Calculation for the USA
param5, covar5 = opt.curve_fit(
    exponential, df_forest_data['Years'], df_forest_data['United States'],
    p0=(1.2e12, 0.03))

print("US Forest area (% of land area)", param5[0]/1e9)
print("Growth rate", param5[1])


df_forest_data["fit_Brazil"] = exponential(df_forest_data['Years'], *param1)
df_forest_data["fit_Canada"] = exponential(df_forest_data['Years'], *param2)
df_forest_data["fit_China"] = exponential(df_forest_data['Years'], *param3)
df_forest_data["fit_United Kingdom"] = exponential(df_forest_data['Years'], *param4)
df_forest_data["fit_United States"] = exponential(df_forest_data['Years'], *param5)
print(df_forest_data)




# Create a line plot of the arable land over time, with the fitted line
# Create a line plot of the arable land over time
plt.figure(figsize=(10.0, 6.0))
plt.plot(df_forest_data['Years'], df_forest_data['Brazil'], marker="o")
plt.plot(df_forest_data["Years"], df_forest_data["fit_Brazil"], linestyle="--")
plt.plot(df_forest_data['Years'], df_forest_data['Canada'], marker="o")
plt.plot(df_forest_data["Years"], df_forest_data["fit_Canada"], linestyle="--")
plt.plot(df_forest_data['Years'], df_forest_data['China'], marker="o")
plt.plot(df_forest_data["Years"], df_forest_data["fit_China"], linestyle="--")
plt.plot(df_forest_data['Years'], df_forest_data['United Kingdom'], marker="o")
plt.plot(df_forest_data["Years"], df_forest_data["fit_United Kingdom"], linestyle="--")
plt.plot(df_forest_data['Years'], df_forest_data['United States'], marker="o")
plt.plot(df_forest_data["Years"], df_forest_data["fit_United States"], linestyle="--")

#plt.plot(df_arable["Year"], df_arable["fit"], linestyle="--", label="Trendline")

# Add a title and axis labels
plt.title("Forest area (% of land area), (Exponential) 1990-2020", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Forest area (% of land area)", fontsize=12)
#Add legend to explain the each lineplot 
plt.legend(['Brazil', 'Canada', 'China', 'United Kingdom', 'United States'], 
           loc='center left', bbox_to_anchor=(1, 0.5))
# Show the plot
plt.show()


# defining logistic function
def logistic(t, n0, g, t0):
    """
    Calculates the logistic function with scale factor n0 and growth rate g
    """

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f

plt.figure(figsize=(10.0, 6.0))
#Calculation for Brazil
param11, covar11 = opt.curve_fit(logistic, df_forest_data['Years'], df_forest_data['Brazil'], p0=(1, 0.03, 1990.0))

sigma11 = np.sqrt(np.diag(covar11))

df_forest_data["fit_log_Brazil"] = logistic(df_forest_data['Years'], *param11)

# Create a line plot of the Forest area (% of land area) over time, with the fitted line

df_forest_data.plot(x="Years", y=['Brazil', "fit_log_Brazil"], 
               ax=plt.gca(), marker="o")

#Calculation for Canada
param22, covar22 = opt.curve_fit(logistic, df_forest_data['Years'], df_forest_data['Canada'], p0=(1, 0.03, 1990.0))

sigma22 = np.sqrt(np.diag(covar22))

df_forest_data["fit_log_Canada"] = logistic(df_forest_data['Years'], *param22)

# Create a line plot of the Forest area (% of land area) over time, with the fitted line
df_forest_data.plot(x="Years", y=['Canada', "fit_log_Canada"], 
               ax=plt.gca(), marker="o")

#Calculation for China
param33, covar33 = opt.curve_fit(logistic, df_forest_data['Years'], df_forest_data['China'], p0=(1, 0.03, 1990.0))

sigma33 = np.sqrt(np.diag(covar33))

df_forest_data["fit_log_China"] = logistic(df_forest_data['Years'], *param33)

# Create a line plot of the Forest area (% of land area) over time, with the fitted line
df_forest_data.plot(x="Years", y=['China', "fit_log_China"], 
               ax=plt.gca(), marker="o")

#Calculation for United Kingdom
param44, covar44 = opt.curve_fit(logistic, df_forest_data['Years'], df_forest_data['United Kingdom'], p0=(1, 0.03, 1990.0))

sigma44 = np.sqrt(np.diag(covar44))

df_forest_data["fit_log_United Kingdom"] = logistic(df_forest_data['Years'], *param44)

# Create a line plot of the Forest area (% of land area) over time, with the fitted line
df_forest_data.plot(x="Years", y=['United Kingdom', "fit_log_United Kingdom"], 
               ax=plt.gca(), marker="o")


#Calculation for US
param55, covar55 = opt.curve_fit(logistic, df_forest_data['Years'], df_forest_data['United States'], p0=(1, 0.03, 1990.0))

sigma55 = np.sqrt(np.diag(covar55))

df_forest_data["fit_log_United States"] = logistic(df_forest_data['Years'], *param55)

# Create a line plot of the Forest area (% of land area) over time, with the fitted line
df_forest_data.plot(x="Years", y=['United States', "fit_log_United States"], 
               ax=plt.gca(), marker="o")

# Add a title and axis labels
plt.title("Forest area (% of land area), (Logistic) 1990-2020", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Forest area (% of land area))", fontsize=12)

# Set the x-axis tick marks to every 5 years
#plt.xticks(range(1960, 2021, 5))

# Add a legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()


print("turning point", param11[2], "+/-", sigma11[2])
print("Forest area (% of land area) at turning point",
      param11[0]/1e9, "+/-", sigma11[0]/1e9)
print("growth rate", param11[1], "+/-", sigma11[1])

print("turning point", param22[2], "+/-", sigma22[2])
print("Forest area (% of land area) at turning point",
      param22[0]/1e9, "+/-", sigma22[0]/1e9)
print("growth rate", param22[1], "+/-", sigma22[1])


print("turning point", param33[2], "+/-", sigma33[2])
print("Forest area (% of land area) at turning point",
      param33[0]/1e9, "+/-", sigma33[0]/1e9)
print("growth rate", param33[1], "+/-", sigma33[1])

print("turning point", param44[2], "+/-", sigma44[2])
print("Forest area (% of land area) at turning point",
      param44[0]/1e9, "+/-", sigma44[0]/1e9)
print("growth rate", param44[1], "+/-", sigma44[1])

print("turning point", param55[2], "+/-", sigma55[2])
print("Forest area (% of land area) at turning point",
      param55[0]/1e9, "+/-", sigma55[0]/1e9)
print("growth rate", param55[1], "+/-", sigma55[1])


print()
df_forcast_forest_data = pd.DataFrame({'Years':[1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]})
print(df_forcast_forest_data["Years"])


# Create a line plot of the Forest area (% of land area) over time, with the forecasted line

df_forcast_forest_data['forecast_Brazil'] = logistic(df_forcast_forest_data["Years"], *param11)
df_forcast_forest_data['forecast_Canada'] = logistic(df_forcast_forest_data["Years"], *param22)
df_forcast_forest_data['forecast_China'] = logistic(df_forcast_forest_data["Years"], *param33)
df_forcast_forest_data['forecast_United Kingdom'] = logistic(df_forcast_forest_data["Years"], *param44)
df_forcast_forest_data['forecast_United States'] = logistic(df_forcast_forest_data["Years"], *param55)
plt.figure(figsize=(10.0, 6.0))
#plt.plot(df_forest_data['Years'], df_forest_data["Brazil"], label="Forest area (% of land area)")
plt.plot(df_forcast_forest_data["Years"], df_forcast_forest_data['forecast_Brazil'])
plt.plot(df_forcast_forest_data["Years"], df_forcast_forest_data['forecast_Canada'])
plt.plot(df_forcast_forest_data["Years"], df_forcast_forest_data['forecast_China'])
plt.plot(df_forcast_forest_data["Years"], df_forcast_forest_data['forecast_United Kingdom'])
plt.plot(df_forcast_forest_data["Years"], df_forcast_forest_data['forecast_United States'])
# Add a title and axis labels
plt.title("Forest area Forecast (% of land area), 1990-2030", fontsize=16)
plt.xlabel("Years", fontsize=12)
plt.ylabel("Forest area (% of land area)", fontsize=12)



# Add a legend
plt.legend(['Brazil', 'Canada', 'China', 'United Kingdom', 'United States'], loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()





'''
# Create a line plot of the arable land over time, with the forecasted line
# and error bands

df_forcast_forest_data["forecast_Brazil"] = poly(df_forcast_forest_data["Years"], param11)
low11, up11 = err.err_ranges(df_forcast_forest_data["Years"], df_forcast_forest_data["forecast_Brazil"], param11, sigma11)

plt.figure(figsize=(10.0, 6.0))
plt.plot(df_forcast_forest_data["Year"], df_forcast_forest_data["forecast_Brazil"])

# Add the error bands to the plot
plt.fill_between(df_forcast_forest_data["Year"], low11, up11, color="yellow", alpha=0.7)

# Add a title and axis labels
plt.title("Arable Land as a Percentage of Land Area, 1960-2030", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Arable Land (% of Land Area)", fontsize=12)

# Set the x-axis tick marks to every 10 years
#plt.xticks(range(1960, 2031, 10))

# Add a legend
plt.legend(fontsize=12)

# Show the plot
plt.show()
'''