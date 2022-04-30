import pandas as pd
import numpy as np


def load_data(filepath, headers=None):
    """ This function helps you load Csv_data
    .Outputs are : x, y, n, m"""

    # Step 1: Read csv_file by the filepath and add names as headers to the data with the help of pandas package
    data = pd.read_csv(filepath,  names=headers)

    # Step 2 : Several determinations to get useful information from our data (ex1data1.csv)

    # n stands for number of features (What s feature? please see this site-->https://www.datarobot.com/wiki/feature/)
    n = data.shape[1]-1

    # m stands for number of training examples
    m = data.shape[0]

    # x stands for all columns except the last one which stands for y
    x = data.values[:, 0:n]

    # Here for y, we had to transpose matrix y with the help of numpy package
    # in order to turn it into a vector m*1 instead of a matrix 1*m
    y = np.transpose(np.array(data.values[:, n])[np.newaxis])

    # Print appropriate details to the user
    print(f'Number of data ={m}')
    print(f'm = Number of training examples ={m}.  Please note that in this program'
          f' all data has been considered as training data')
    print(f'n = Number of features ={n}')
    print('x = the first colum ')
    print('y =  the second colum')
    print('---------------------------------------------------')
    print(data)

    # Step 3: Return certain variables to be used in other sections or other programs
    return x, y, n, m