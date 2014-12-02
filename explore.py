import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np

# load source data
data = pd.read_csv('source/20140129.0.click.0.csv')#, index_col=0, parse_dates=True)
click = pd.DataFrame(data)

data = pd.read_csv('source/20140129.0.conversion.0.csv')#, index_col=0, parse_dates=True)
conv = pd.DataFrame(data)

data = pd.read_csv('source/20140129.0.view.0.csv')#, index_col=0, parse_dates=True)
view = pd.DataFrame(data)

# combine dataframes
full = click.append(conv)
full = full.append(view)

# convert columns to correct datatype
for i in ['event_time','request_time', 'view_time']:
    full[i] = pd.to_datetime(full[i]/1000000, unit='s')
print full.head()
print full.shape