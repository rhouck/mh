import pandas as pd
import numpy as np
import datetime
import csv

# load source data
# index records by auction_id
data = pd.read_csv('source/20140129.0.click.0.csv', index_col=2)
click = pd.DataFrame(data)
 
data = pd.read_csv('source/20140129.0.view.0.csv', index_col=2)
view = pd.DataFrame(data)

# convert unix timestamps to datetime
for i in ['event_time','request_time', 'view_time']:
    view[i] = pd.to_datetime(view[i]/1000000, unit='s')

# add new column to view data signifiying whether an ad was clicked or not
# initialize feature as zeros
view['clicked'] = 0
 
# select any click column and rename it to join with view dataframe
click_series = click['event_type']
click_series.name = 'event_type_click'
 
# change clicked ad rows 'clicked' value to 1
matched = view.join(click_series, how='inner')
view.loc[matched.index, 'clicked'] = 1

# isolate columns that can be manipulated to affect CTR
# ignore line item id
# ignore url - assuming there are too many to make sense of, especially considering how few rows are available
# ignore georgraphy
# ignore time / day of week
cols = ['clicked',
        'creative_id', 
        'universal_site_id', 
        'adx_page_categories', 
        'matching_targeted_keywords', 
        'exchange', 
        'ad_position', 
        'matching_targeted_segments', 
        'device_type'
        ]

# this function returns a sparce matrix containing hashed feature:value pairs for each feature vector (row)
# with online (iterative) learning, it's impossible to know the full range of categorical values for a feature ahead of time
# to get around this, we create new features for each unique feature:value pairs by storing hashed values to represent indices in a large vector
""" 
from sklearn.feature_extraction import FeatureHasher
 
# this is the hashing algorithm we'll be using to map the feature set of unknown complexity
hasher = FeatureHasher(input_type='string', n_features=(2 ** 15))
"""
def build_row(raw_x):
    
    # drop universal site id when adx_page_categories is supplied
    if not isinstance(raw_x['adx_page_categories'], float):
        raw_x['universal_site_id'] = np.nan

    x = []
    for count, value in enumerate(raw_x):
        
        # check for string values which represent a list of multiple values
        # used for adx_page_categories and matching_targeted_segments
        if isinstance(value, str):    
            vals = value.split()
            for v in vals:
                x.append('F%s:%s' % (count,v))
            continue
        
        if  np.isnan(value):
            continue
        
        x.append('F%s:%s' % (count,value))
    
    return x

rows = []
ind_bank = []
targets = []
full_length = view.shape[0]

with open('source/ys_prepped.csv', 'wb') as ys, open('source/xs_prepped.csv', 'wb') as xs:
	ys_writer = csv.writer(ys)
	xs_writer = csv.writer(xs)

	# convert data set to bank array of sparce matrices
	for t, i in enumerate(view.index):

		# add new row to bank
		y = view.ix[i][cols[0]]
		raw_x = view.ix[i][cols[1:]]

		row = build_row(raw_x,)

		# create bank to store formatted rows
		# store recent indexes to drop from data frame in batches
		targets.append(str(y))
		rows.append(row)
		ind_bank.append(i)

		# monitor progress
		if ((t + 1) % 20000 == 0) or ((t + 1) == full_length):   
			ys_writer.writerows(targets)
			xs_writer.writerows(rows)
			targets = []
			rows = []
			
			# drop rows from data frame
			view = view.drop(ind_bank)
			ind_bank = []   
			print '%s\trows processed: %d' % (datetime.datetime.now(), t+1)

	view = view.drop(ind_bank)
	print "Finished formatting rows for conversion to sparse matrices"

