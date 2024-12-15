import pandas as pd

def filter_records(housing):
    '''
    Filter data so records with NA values of specific values are removed 
    '''
    df = pd.read_csv(housing) 

    # List of columns to check for NA values
    columns_to_check = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms', 'history_price']

    # Drop rows with NA in any of the specified columns 
    cleaned_df = df.dropna(subset=columns_to_check)

    # save cleaned dataset to our folder 
    cleaned_df.to_csv('cleaned_london_house_price_data.csv', index=False)
    

if __name__ == "__main__":

    housing_file = 'london_house_price_data.csv' 
    filter_records(housing_file)
    
