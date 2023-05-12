import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Start of 1st program funtion Forest area (% of land area) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_forest = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", header=2, index_col='Country Name', usecols=[
                          'Country Name', 'Indicator Name', '1995', '2000', '2010', '2020'])


print(data_forest)
selected_countries = ['Brazil', 'United States', 'China', 'United Kingdom',
                      'Australia', 'Russian Federation', 'South Africa', 'Canada', 'India', 'Spain']

selected_data_forest = data_forest.loc[selected_countries]
print(selected_data_forest)

# transpose the dataframe
data_forest_transposed = selected_data_forest.transpose()


print(data_forest_transposed)

data_forest_t_cleaned = data_forest_transposed.dropna()

# set the year column as an integer

data_forest_t_cleaned.iloc[1:,
                           :] = data_forest_t_cleaned.iloc[1:, :].astype(int)

print(data_forest_t_cleaned.describe())

print(data_forest_t_cleaned.iloc[1:, :])

# plot the bar chart
chart1 = data_forest_t_cleaned.iloc[1:, :].plot(
    kind='bar', stacked=False, colormap='rainbow')
# labelling x axis and y axis
chart1.set_xlabel('Year')
chart1.set_ylabel('Forest area (% of land area)')
# Add Title to the graph
chart1.set_title('Forest area (% of land area) 1995 to 2020')
# Add legend to explain the each lineplot
chart1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# save the plot output as png
plt.savefig('Forest area (% of land area).png')
# Display the plot
plt.show()


# Start of 2nd program funtion Average precipitation in depth (mm per year) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_precipitation = pd.read_csv("API_AG.LND.PRCP.MM_DS2_en_csv_v2_5456478.csv", header=2, index_col='Country Name', usecols=[
                                 'Country Name', 'Indicator Name', '1995', '2000', '2010', '2020'])


print(data_precipitation)
selected_countries = ['Brazil', 'United States', 'China', 'United Kingdom',
                      'Australia', 'Russian Federation', 'South Africa', 'Canada', 'India', 'Spain']

selected_data_precipitation = data_precipitation.loc[selected_countries]
print(selected_data_precipitation)

# transpose the dataframe
data_precipitation_transposed = selected_data_precipitation.transpose()


print(data_precipitation_transposed)

data_precipitation_t_cleaned = data_precipitation_transposed.dropna()

# set the year column as an integer

data_precipitation_t_cleaned.iloc[1:,
                                  :] = data_precipitation_t_cleaned.iloc[1:, :].astype(int)

print(data_precipitation_t_cleaned.describe())

print(data_precipitation_t_cleaned.iloc[1:, :])

# plot the bar chart
chart1 = data_precipitation_t_cleaned.iloc[1:, :].plot(
    kind='bar', stacked=False, colormap='rainbow')
# labelling x axis and y axis
chart1.set_xlabel('Year')
chart1.set_ylabel('Average precipitation in depth (mm per year)')
# Add Title to the graph
chart1.set_title('Average precipitation in depth (mm per year) 1995 to 2020')
# Add legend to explain the each lineplot
chart1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# save the plot output as png
plt.savefig('Average precipitation in depth (mm per year).png')
# Display the plot
plt.show()

# Start of 3rd program funtion Average precipitation in depth (mm per year) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_CO2 = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5455005.csv", header=2, index_col='Country Name', usecols=[
                                 'Country Name', 'Indicator Name', '1995', '2000', '2010', '2019'])


print(data_CO2)
selected_countries = ['Brazil', 'United States', 'China', 'United Kingdom',
                      'Australia', 'Russian Federation', 'South Africa', 'Canada', 'India', 'Spain']

selected_data_CO2 = data_CO2.loc[selected_countries]
print(selected_data_CO2)

# transpose the dataframe
data_CO2_transposed = selected_data_CO2.transpose()


print(data_CO2_transposed)

data_CO2_t_cleaned = data_CO2_transposed.dropna()

# set the year column as an integer

data_CO2_t_cleaned.iloc[1:,
                                  :] = data_CO2_t_cleaned.iloc[1:, :].astype(int)

print(data_CO2_t_cleaned.describe())

print(data_CO2_t_cleaned.iloc[1:, :])

# plot the bar chart
chart1 = data_CO2_t_cleaned.iloc[1:, :].plot(
    kind='bar', stacked=False, colormap='rainbow')
# labelling x axis and y axis
chart1.set_xlabel('Year')
chart1.set_ylabel('CO2 emissions (kt)')
# Add Title to the graph
chart1.set_title('CO2 emissions (kt) 1995 to 2020')
# Add legend to explain the each lineplot
chart1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# save the plot output as png
plt.savefig('CO2 emissions (kt).png')
# Display the plot
plt.show()

# Start of 4th program funtion Average precipitation in depth (mm per year) from 1995 to 2020
# Reading data from the source file. Here source file type is csv.
# Therefore pandas reading funtion must be pd.read_csv
data_population = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_5454896.csv", header=2, index_col='Country Name', usecols=[
                                 'Country Name', 'Indicator Name', '1995', '2000', '2010', '2020'])


print(data_precipitation)
selected_countries = ['Brazil', 'United States', 'China', 'United Kingdom',
                      'Australia', 'Russian Federation', 'South Africa', 'Canada', 'India', 'Spain']

selected_data_population = data_population.loc[selected_countries]
print(selected_data_population)

# transpose the dataframe
data_population_transposed = selected_data_population.transpose()


print(data_population_transposed)

data_population_t_cleaned = data_population_transposed.dropna()

# set the year column as an integer

data_population_t_cleaned.iloc[1:, :] = data_population_t_cleaned.iloc[1:, :].astype(int)

print(data_population_t_cleaned.describe())

print(data_population_t_cleaned.iloc[1:, :])

# plot the bar chart
chart1 = data_population_t_cleaned.iloc[1:, :].plot(
    kind='bar', stacked=False, colormap='rainbow')
# labelling x axis and y axis
chart1.set_xlabel('Year')
chart1.set_ylabel('Population Total')
# Add Title to the graph
chart1.set_title('Population Total 1995 to 2020')
# Add legend to explain the each lineplot
chart1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# save the plot output as png
plt.savefig('Population Total.png')
# Display the plot
plt.show()

# Plot the data
plt.scatter(data_forest_t_cleaned.iloc[1:, :], data_precipitation_t_cleaned.iloc[1:, :], color='green', label='Data Set 1')

# Repeat for other data sets
plt.scatter(data_population_t_cleaned.iloc[1:, :], data_precipitation_t_cleaned.iloc[1:, :], color='blue', label='Data Set 2')

# Repeat for other data sets
plt.scatter(data_population_t_cleaned.iloc[1:, :], data_CO2_t_cleaned.iloc[1:, :], color='red', label='Data Set 2')
# Add labels and legend
plt.ylabel('Precipitation, Forest Area, CO2 emission')
plt.xlabel('Population Total')
#plt.legend()

# Show the plot
plt.show()



combined_data = pd.merge(data_population_t_cleaned.iloc[1:, :], data_forest_t_cleaned.iloc[1:, :], on=['Country Name'])
combined_data = pd.merge(combined_data, data_precipitation_t_cleaned.iloc[1:, :], on=['Country Name'])
combined_data = pd.merge(combined_data, data_CO2_t_cleaned.iloc[1:, :], on=['Country Name'])
print(combined_data)

sns.heatmap(combined_data, cmap='YlGnBu')
plt.title('Heatmap of Population, Forest Area, Precipitation, and CO2 Emission')
plt.show()