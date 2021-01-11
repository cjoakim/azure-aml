# This script is executed as a "Execute Python Script" in Azure Machine Learning Designer.
# The script MUST contain an entry point function named .
# 'azureml_main' MUST have two input arguments, both are pandas.DataFrame objects.
#   Param<dataframe1>: a pandas.DataFrame  (port1)
#   Param<dataframe2>: a pandas.DataFrame  (port2)
# If the input port is not connected, the corresponding dataframe argument will be None.
#
# The return value must be of a sequence of one or two pandas.DataFrame objects.
#   Single return value: return dataframe1,
#   Two return values: return dataframe1, dataframe2


import pandas as pd


def azureml_main(dataframe1 = None, dataframe2 = None):
    print_df_info(dataframe1, 'dataframe1')
    print_df_info(dataframe2, 'dataframe2')

    # Simply convert the state column to uppercase
    dataframe1['state'] = dataframe1['state'].str.upper()

    print_df_info(dataframe1, 'output df')

    return dataframe1,

def print_df_info(df, msg=''):
    print('')
    print('print_df_info: {}'.format(msg))
    try:
        print(df.head(3))
        print(df.tail(3))
        print(df.dtypes)
        print(df.shape)
        print(df.columns)
    except:
        print('df is None or invalid')
