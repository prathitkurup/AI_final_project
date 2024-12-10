#Use Geopy to convert our addresses from the dataset into coordinates that we can use to map regions
from geopy.geocoders import Nominatim
import pandas as pd

df = pd.read_csv('rollingsales_manhattan.csv')
sample_address = df.ADDRESS[1000:1005]

geolocator = Nominatim(user_agent="Develop_project")
for address in sample_address:
    location = geolocator.geocode(f"{address.split(',')} manhattan")
    print(f"1: {location}")

