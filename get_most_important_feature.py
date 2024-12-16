import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def get_most_important_feature(housing_df: pd.DataFrame, features) -> str:
    """
    Scales the data, fits a linear regression model, and outputs the coefficient with highest magnitude (relating to its importance)
    """
    
    #Creates a list of the features we want to model as inputs
    inputs = housing_df[features]
    target = housing_df.history_price

    #create a scaler object and standardize the data
    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(inputs)
    
    #run multiple regression on the standardized data
    model = LinearRegression()
    model.fit(inputs_scaled, target)

    return pd.DataFrame({'features': features,
                         'coefficient': model.coef_})

# Load the cleaned London house price data
df = pd.read_csv('cleaned_london_house_price_data.csv')

# Limit data for faster computation
df = df.iloc[:70000]

#define important features
features = ['latitude','longitude', 'bathrooms','bedrooms','floorAreaSqM', 'livingRooms']

# Drop any rows with NaN values in the relevant columns
df = df.dropna()

print(get_most_important_feature(df, features))