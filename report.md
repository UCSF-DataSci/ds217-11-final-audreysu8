# Chicago Beach Weather Sensors Dataset Explorations

## Executive Summary

For this analysis, I investigate data from weather sensors from beaches in Chicago, Illinois, USA along the Lake Michigan lakefront. The weather sensor data contains 196,313 observations recorded hourly from three weather stations spanning April 25, 2015 to December 3, 2025.

The goal of the project was to follow the 9-phase data science workflow to examine temporal trends in weather in Chicago's beachses along Lake Michigan as well as apply machine learning models to the Chicago beach weather sensors data to generate predictions for Air Temperature (°C). 

The key finding of my work was that Random Forest Regressor emerged as the top-predicting model for Air Temperature (°C) with an R² Score of 0.736717, RMSE of 26.997480 °C, and MAE of 3.743533 °C; however, since RMSE and MAE are fairly large, I would say that further work is needed to improve air temperature predictions from beach sensor data. 

## Phase-by-Phase Findings

### Phase 1-2: Exploration

After conducting some initial explorations of the Chicago beach weather sensors dataset, I found that the dataset contains 196,313 rows and 18 columns. The dataset contains features such as "Station Name", "Measurement Timestamp", "Air Temperature", "Wet Bulb Temperature", "Precipitation Type", "Wind Speed", and "Solar Radiation". "Station Name" corresponds to the name weather sensor station that measured a given data entry and "Measurement Timestamp" indicates the time (specific to the hour) at which a given data entry was recorded. weather The data was measured from April 25, 2015 at 9 AM to December 3, 2025 at 10 AM. 

All of the features of the dataset save for "Station Name", "Measurement Timestamp", "Measurement Timestamp Label", and "Measurement ID" are numeric. Below is a table providing more detailed about the type of each feature of the data:

| Column Name                  | Data Type         |
|------------------------------|-------------------|
| Station Name                 | object            |
| Measurement Timestamp        | object            |
| Air Temperature              | float64           |
| Wet Bulb Temperature         | float64           |
| Humidity                     | int64             |
| Rain Intensity               | float64           |
| Interval Rain                | float64           |
| Total Rain                   | float64           |
| Precipitation Type           | float64           |
| Wind Direction               | int64             |
| Wind Speed                   | float64           |
| Maximum Wind Speed           | float64           |
| Barometric Pressure          | float64           |
| Solar Radiation              | int64             |
| Heading                      | float64           |
| Battery Life                 | float64           |
| Measurement Timestamp Label  | object            |
| Measurement ID               | object            |

Below is a table displaying some summary statistics of the numeric features of the dataset: 

| column_name          | mean                   | std                      | min       | max     | missing_count |
|----------------------|------------------------|---------------------------|-----------|---------|----------------|
| Air Temperature      | 12.624225073635074     | 10.435491250453873        | -29.78    | 37.6    | 75             |
| Wet Bulb Temperature | 10.27482096273034      | 9.403981662014425         | -28.9     | 28.4    | 75947          |
| Humidity             | 68.02391588942149      | 15.633995713561301        | 0.0       | 100.0   | 0              |
| Rain Intensity       | 0.15892693950118802    | 1.794022816610356         | 0.0       | 183.6   | 75947          |
| Interval Rain        | 0.14236790227850427    | 1.0969230607538099        | -0.9      | 63.42   | 0              |
| Total Rain           | 141.48457122443216     | 190.4590771816808         | 0.0       | 1056.1  | 75947          |
| Precipitation Type   | 4.267774953059834      | 15.589548562681943        | 0.0       | 70.0    | 75947          |
| Wind Direction       | 140.80295752191654     | 122.00747357163634        | 0.0       | 359.0   | 0              |
| Wind Speed           | 2.918785307137072      | 5.341879881583158         | 0.0       | 999.9   | 0              |
| Maximum Wind Speed   | 3.556968208931655      | 5.955099747225225         | 0.0       | 999.9   | 0              |
| Barometric Pressure  | 994.3134273348728      | 10.02922023732468         | 0.0       | 3098.5  | 146            |
| Solar Radiation      | 112.34736874277301     | 842.8080702318339         | -100000.0 | 1277.0  | 0              |
| Heading              | 281.9668344881445      | 142.7717524102073         | 0.0       | 359.0   | 75947          |
| Battery Life         | 13.163250014008243     | 1.5446151630769804        | 0.0       | 15.3    | 0              |

I generalized some initial visualizations of the distribution of Air Temperature (°C) and Web Bulb Temperature (°C) over time. Below are the visualizations:
![Figure 1: Initial Visualizations](output/q1_visualizations.png)
* Figure 1: The plot on the left displays the distribution of Air Temperature (°C) in the Chicago beach weather sensors dataset and the plot on the right displays a time series plot of Web Bulb Temperature (°C)

From these visualizations, one can see that Air Temperature ranges from around -30 °C to around 38 °C, with two peaks that occur at around 2-4 °C and 20-24 °C. One can also observe that Web Bulb Temperature follows seasonal patterns. 

**Key Data Quality Issues:**
I identified some important data quality issues in the dataset. The dataset contains 7 columns with missing data: Air Temperature, Wet Bulb Temperature, Rain Intensity, Total Rain, Precipitation Type, Barometric Pressure, and Heading. Here is a more detailed breakdown of the number of missing values and the percentage of values misssing contained in these columns: 

| Column Name                  | Number of Missing Values   | Percentage Missing Values
|------------------------------|----------------------------|-------------------
| Air Temperature              | 75                         | approximately 0%
| Wet Bulb Temperature         | 75947                      | 0.4%
| Rain Intensity               | 75947                      | 0.4%
| Total Rain                   | 75947                      | 0.4%
| Precipitation Type           | 75947                      | 0.4%
| Barometric Pressure          | 146                        | approximately 0%
| Heading                      | 75947                      | 0.4%

I defined an outlier as a value having a z-score with a magnitude greater than 3. I found that columns of Air Temperature, Wet Bulb Temperature, Humidity, Rain Intensity, Interval Rain, Total Rain, Precipitation Type, Wind Speed, Maximum Wind Speed, Barometric Pressure, Solar Radiation, and Battery Life contain outliers. Here is a more detailed breakdown of the columns with outliers and the corresponding number of outliers the column has:

| Column                | Count |
|-----------------------|-------|
| Air Temperature       | 296   |
| Wet Bulb Temperature  | 487   |
| Humidity              | 134   |
| Rain Intensity        | 1434  |
| Interval Rain         | 2298  |
| Total Rain            | 4023  |
| Precipitation Type    | 12657 |
| Wind Speed            | 27    |
| Maximum Wind Speed    | 20    |
| Barometric Pressure   | 87    |
| Solar Radiation       | 13    |
| Battery Life          | 6     |

I also noticed that there were some hours between the start time of the dataset at April 25, 2015 at 9 AM to December 3, 2025 at 10 AM where data wasn't recorded. 

### Phase 3: Data Cleaning

I performed data cleaning on the Chicago beach weather sensors dataset to handle missing data and outliers, validate data types, and remove duplicates so that the data is in a more reliable format for future analysis that I will be conducting. 

An interesting observation is that all of the columns that contained missing values and all of the columns that contained outliers were numeric columns. I decided to impute misisng values in using forward-fill in order to better maintain overall trends that our time series data contains. I defined an outlier as a value with z-score with a magnitude greater than 3, and capped outliers to have at most a z-score with a magnitude of 3. By imputing missing values and capping outliers, we are able to retain our previous dataset size and can ensure that the dataset doesn't lose a significant amount of entries before we carry out later steps in the data science workflow. 

**Cleaning Results:**
- Number of rows before cleaning: 196,313
- Missing Data Handling: all missing values were imputed using forward-fill 
  - Air Temperature: 75 missing values handled
  - Wet Bulb Temperature: 75947 missing values handled
  - Rain Intensity: 75947 missing values handled
  - Total Rain: 75947 missing values handled
  - Precipitation Type: 75947 missing values handled 
  - Barometric Pressure: 146 missing values handled
  - Heading: 75947 missing values handled
- Outlier Handling: capped outliers to have at most a z-score of magnitude 3 (3 standard deviations from the mean)
  - Air Temperature: 296 outliers handled
  - Wet Bulb Temperature: 487 outliers handled
  - Humidity: 134 outliers handled 
  - Rain Intensity: 1434 outliers handled 
  - Interval Rain: 2298 outliers handled 
  - Total Rain: 4023 outliers handled 
  - Precipitation Type: 12657 outliers handled 
  - Wind Speed: 27 outliers handled 
  - Maximum Wind Speed: 20 outliers handled 
  - Barometric Pressure: 87 outliers handled 
  - Solar Radiation: 13 outliers handled 
  - Battery Life: 6 outliers handled
- Number of duplicate rows removed: 0
- Data Type Conversions:
  - 'Measurement Timestamp' converted to datetime64[ns] format
- Number of rows after cleaning: 196313

The data cleaning I applied on the dataset was able to keep the dataset size the same as before the same while also address data quality issues. 

### Phase 4: Data Wrangling

I parsed the "Measurement Timestamp" column from its original ‘MM/DD/YYYY HH:MM:SS AM/PM’ string format and converted it into a datetime object. I then set the "Measurement Timestamp" column as the index of the DataFrame so that I can conduct more time series operations on the data. I also derived additional temporal features from Measurement Timestamp to support further temporal analysis. Here is a more detailed breakdown of the new temporal features I created: 

- `hour`: hour of the day using 24-hour time (0 corresponds to midnight and 23:00 correspond to 11 PM)
- `day_of_week`: day of the week (0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday)
- `month`: month of the year (represented in number form, for example 1 = January)
- `year`: year
- `day_name`: name of the day (e.g. Monday)
- `is_weekend`: binary indicator of whether it is weekend or not (1 if weekend, 0 if weekday)

### Phase 5: Feature Engineering

I applied feature engineering on the dataset to extract additional variables to transform the original variables in the data into more informative variables so that machine learning models applied on the data can better capture the data's underlying patterns. I also decided to create rolling window features to better represent time-based interactions in the data. Here is a more detailed breakdown of the additional features I created: 

- `Wind Speed Squared`: values in `Wind Speed` column squared 
- Is Raining: binary indicator of whether it is raining or not (1 if raining, 0 if not raining)
- Is Summer: binary indicator of whether data was recorded during summer or not (1 if it is summer, 0 if it is not summer)
- Is Winter: binary indicator of whether data was recorded during winter or not (1 if it is winter, 0 if it is not winter)
- Is Spring: binary indicator of whether data was recorded during spring or not (1 if it is spring, 0 if it is not spring)
- Is Fall: binary indicator of whether data was recorded during fall or not (1 if it is spring, 0 if it is not fall)
- Maximum Wind Speed - Wind Speed: difference between the maximum wind speed recorded duing the hour in mph and the average wind speed recorded during the hour in mph 
- Wind Direction X: x-coordinate of wind direction, calculated by \[
\text{Wind Direction X} = \text{Wind Speed}  \cdot \cos\left(\text{Wind Direction} \cdot \frac{\pi}{180}\right)
\]
- Wind Direction Y: y-coordinate of wind direction, calculated by \[
\text{Wind Direction Y} = \text{Wind Speed}  \cdot \sin\left(\text{Wind Direction} \cdot \frac{\pi}{180}\right)
\]
- Rain Intensity x Humidity: captures the interaction betwen rain intensity and humidity, calculated by multiplying values in Rain Intensity column by  corresponding value in same row in Humidity column
- Sin_Hour: sine transform of hour in day in order to hour of day into a cyclical value, calculated by \[
\text{Sin_Hour} =  \sin\left( 2 \pi \frac{\text{hour}}{24} \right)
\]
- Cos_Hour: cosine transform of hour in day in order to hour of day into a cyclical value, calculated by \[
\text{Cos_Hour} =  \cos\left( 2 \pi \frac{\text{hour}}{24} \right)
\]

**Rolling Window Features:**

- barometric_pressure_rolling_mean_7h: 7 hour rolling mean of Barometric Pressure
- humidity_rolling_mean_7h: 7 hour rolling mean of Humidity
- solar_radiation_rolling_mean_7h: 7 hour rolling mean of Solar Radiation
- total_rain_rolling_mean_24h: 7 hour rolling mean of Total Rain

Note: I did not extract any new features from "Air Temperature", as that is the variable I am trying to predict with my analysis. Feeding new features extracted from "Air Temperature" into the machine learning models would cause data leakage, as I would be essentially informing the machine learning models of the results during training. 

### Phase 6: Pattern Analysis

I then applied further pattern analysis and advanced visualizations to the data.

Here are some interesting visualizations that I generated: 

![Figure 2: Pattern Analysis](output/q5_patterns.png)
* Figure 1: The plot on the left displays monthly trends in solar radiation (watts per square meter) and the plot on the right displays daily patterns in Wind Speed (mph) and Humidity (%) by hour of day

**Temporal Trends**:
- Monthly average solar radiation shows seasonal pattern
- Higher solar radiation in summer months (June to August)
- Lower solar radiation in winter months (December to February)
- Monthly solar radiation range: 24.5 to 238.3 watts per square meter

**Daily Patterns:**
 - Wind speed tends to be higher from around 9 AM to 6 PM
 - Wind speed achieves peak at around 1 PM
 - Minimum wind speed occurs at midnight
 - Humidity tends to be higher from midnight to 9 AM
 - Humidity achieves peak at around 6 AM
 - Minimum humidity occurs at around 4 PM

**Correlations:**
- Air Temp vs Water Temp: 0.83 (strong positive correlation)
- Air Temp vs Is Summer: 0.62 (moderate positive correlation)

### Phase 7: Modeling Preparation

The next step was to prepare the dataset to be fed into the machine learning models later. I selected which features to feed into the machine learning models and the target variable. Then, I performed a temporal train/test split on the data. 

I made "Air Temperature" the target variable with the goal of investigating whether other measured conditions at the Chicago beaches can be used to predict "Air Temperature" or not. 

**Feature Selection:**

I excluded 'Station Name' from being a predictor because I felt that which beach weather sensor station recorded the data should not affect the temperature significntly. I excluded 'Measurement ID' and 'Measurement Timestamp Label' because they were metadata variables that described more information about the Measurement Timestamp column. I also removed 'Wet Bulb Temperature' because it is a very similar measurement to 'Air Temperature', and I wanted to prevent data leakage.

From the remaining features, I calculated their correlations with the target variable "Air Temperature", and selected the features with the top 20 highest correlations to be used for the futuer modeling phase. This was done to ensure that only features with high predictive power are fed into the machine learning models while minimizing model complexity to reduce overfitting. 

Features chosen: 'Is Summer', 'total_rain_rolling_mean_24h', 'Total Rain', 'solar_radiation_rolling_mean_7h', 'month', 'Solar Radiation', 'Is Fall', 'Wind Direction Y', 'hour', 'humidity_rolling_mean_7h', 'Heading', 'Interval Rain', 'Humidity', 'Maximum Wind Speed - Wind Speed', 'year', 'Rain Intensity x Humidity', 'Rain Intensity', 'is_weekend', 'day_of_week', 'Cos_Hour'

The dataset was of dimensions 196313 rows × 18 columns before temporal train/test split

**Temporal Train/Test Split:**
- Split Method: Temporal (80/20 split by time)
- Training Set Size: 157049
- Training Date Range: 2015-04-25 09:00:00 to 2023-07-05 22:00:00
- Test Set Size: 39264
- Test Date Range: 2023-07-05 23:00:00 to 2023-07-05 22:00:00

The reason why I use a temporal train/test split is to prevent datapoints in the future from being in the training data, as that would cause data leakage. 

### Phase 8: Modeling

I applied two models: Linear Regression and Random Forest Regressor on the data. Below is a table comparing the performance of the two models across the training set and test set using error metrics of R², RMSE, and MAE: 

**Model Performance:**

| Model                   | Metric       | Train           | Test           |
|-------------------------|-------------|----------------|----------------|
| Linear Regression       | R²          | 0.60498        | 0.56953        |
|                         | RMSE        | 6.59300        | 6.64386        |
|                         | MAE         | 4.97993        | 5.17897        |
| Random Forest Regressor | R²          | 0.88826        | 0.73672        |
|                         | RMSE        | 3.50659        | 5.19591        |
|                         | MAE         | 2.57324        | 3.74353        |

Below are the features in the Random Forest Regressor model ranked by feature importance 

**Feature Importance (Random Forest Regressor):**
Top features by importance:
1.  Is Summer (0.4453 importance)
2. month (0.3862 importance)
3. total_rain_rolling_mean_24h (0.0518 importance)
4. solar_radiation_rolling_mean_7h (0.0284 importance)
5. Humidity (0.0206 importance)
6. humidity_rolling_mean_7h (0.0181 importance)
7. year (0.0167 importance)
8. Wind Direction Y (0.0083 importance)
9. day_of_week (0.006 importance)
10. Solar Radiation (0.0056 importance)
11. Total Rain (0.0035 importance)
12. hour (0.0034 importance)
13. Is Fall (0.0025 importance)
14. Heading (0.0014 importance)
15. Maximum Wind Speed - Wind Speed (0.0013 importance)
16. is_weekend (0.0004 importance)
17. Cos_Hour (0.0002 importance)
18. Rain Intensity x Humidity (0.0001 importance)
19. Interval Rain (0.0001 importance)
20. Rain Intensity (0.0 importance)





