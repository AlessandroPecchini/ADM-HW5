import pandas as pd
from datetime import datetime
import numpy as np

def get_csv_from_txt(fpath):
    '''
        Given the path of the txt containing the data, it returns a pandas dataframe that has 
        the dates (till the day) for each entry in format datetime.
        In this function it's added 1221436800 to the timestamp since accordingly to wikipedia en
        it's the release date of the first public beta version of the site
    '''
    print('new')
    df = pd.read_csv(fpath, header=None, sep=' ')
    df.rename(columns={0: 'user_from', 1: 'time', 2:'user_to'}, inplace=True)
    df.time=df.time.apply(lambda t: datetime.fromtimestamp(t+1221436800))
    return df
