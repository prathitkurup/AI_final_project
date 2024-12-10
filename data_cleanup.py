import pandas as pd

def filter_res_units(housing):
    '''
    Filter records with residential units 1 
    '''
    filtered_housing = housing[housing['RESIDENTIAL UNITS'] == 1]
    # num_rows2 = filtered_housing.shape[0]
    # print(f"Number of rows (AFTER): {num_rows2}")
    return filtered_housing

if __name__ == "__main__":

    xlsx_file = 'rollingsales_manhattan.xlsx'  

    housing = pd.read_excel(xlsx_file, header=4)  
    housing.to_csv('rollingsales_manhattan.csv', index=False)
    housing.columns = housing.columns.str.strip()

    # num_rows1 = housing.shape[0]
    # print(f"Number of rows (BEFORE): {num_rows1}")

    filter_res_units(housing)
    
