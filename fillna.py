import pandas as pd
from datetime import timedelta
import numpy as np


# Function to insert row in the dataframe
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0
   
    # End value of upper half
    end_upper = row_number
   
    # Start value of lower half
    start_lower = row_number
   
    # End value of lower half
    end_lower = df.shape[0]
   
    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]
   
    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]
   
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
   
    # Combine the two lists
    index_ = upper_half + lower_half
   
    # Update the index of the dataframe
    df.index = index_
   
    # Insert a row at the end
    df.loc[row_number] = row_value
    
    # Sort the index labels
    df = df.sort_index()
   
    # return the dataframe
    return df
   

# добавляет в датафрейм строки с пропущенным часом и np.nan в признаках для дальнейшего заполнениях
data_x = pd.read_csv('data.csv', sep=',')#, index_col=['dt'])
data_x.loc[:,"dt"] = pd.to_datetime(data_x["dt"])
count = 0
for n in range(1, len(data_x)):
    if data_x['dt'][n-1].hour == 10 and data_x['dt'][n].hour == 12:
        row_number = n
        row_value = [np.nan, (data_x['dt'][n-1]) + timedelta(hours=1),
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, data_x['dt'][n].dayofyear, ((data_x['dt'][n-1]) + timedelta(hours=1)).hour]
        data_x = Insert_row(row_number, data_x, row_value)
        count += 1

    
# Заполнение NaN
# data_x = pd.to_datetime(data_x['dt'])
list = ['fact','10_metre_V_wind_component','Snow_density' ,'Snowfall','Visibility','Surface_pressure','Convective_precipitation',
'Visual_cloud_cover','Total_cloud_cover','Precipitation_type','Instantaneous_10_metre_wind_gust','Medium_cloud_cover',
'Total_precipitation_rate','Convective_available_potential_energy','10_metre_U_wind_component','Skin_temperature',
'2_metre_temperature','Surface_solar_radiation_downwards','Wind_speed','Low_cloud_cover','Snow_depth','High_cloud_cover',
'Evaporation','Wind_Direction','2_metre_dewpoint_temperature','Total_precipitation','2_metre_relative_humidity',
'Clear_sky_direct_solar_radiation_at_surface','Snow_height']
# data_x = data_x.drop(['n'], axis=1)
for x in list:
    # data_x[x] = data_x[x].fillna(data_x[x].median())
    data_x[x] = data_x[x].fillna(method='ffill')

# data_x.to_csv("updated_data_full.csv", index=False)
# data_x = pd.read_csv('updated_data_full.csv', sep=',')

# создаем и заполняем список со строками, подлежащими удалению
del_rows = []
for n in range(2, len(data_x)):
    if (data_x['dt'][n].hour !=  data_x['dt'][n-1].hour + 1 and data_x['dt'][n-1].hour != 23 and data_x['dt'][n] != 0):
        p = n - 1
        while data_x['dt'][p].hour != 23:
            del_rows.append(data_x['dt'][p])  
            p-=1
    if (data_x['dt'][n-1].hour == 23 and data_x['dt'][n] != 0):
        p = n
        while data_x['dt'][p].hour != 0:
            del_rows.append(data_x['dt'][p])  
            p+=1

#TODO: 12 часовые обрезки есть. 
# исключаем строки из датафрейма
data_y = data_x.loc[~data_x['dt'].isin(del_rows)]  
data_y.to_csv("data_full.csv", index=False)