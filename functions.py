import pandas as pd
from datetime import datetime
from time import time
import networkx as nx
from tqdm import tqdm
import os

def cast_times(fpath):
    '''
        Given the path of a .txt file in the form of the one of interest
        this functions casts the timestamps in such a way that they refers always to the first hour of the day in order to remove
        the little time differences between timestamps.

        This functions adds also 1221436800 to the timestamp that's the timestamp of the first public beta release of the site
        (accordingly to wikipedia en)

        saves the result into a csv file in the same path adding _casted to the original name 
    '''
    if os.path.exists(fpath[:-4]+'_casted.csv'):
        print(f"the output file: {fpath[:-4]+'_casted.csv'} already exists!")
        return
    df = pd.read_csv(fpath, header=None, sep= ' ')
    df.columns = ['user_from', 'time', 'user_to']
    df.time = df.time.apply(lambda t: t-(t%86400)+1221436800)
    df.sort_values(by='time', inplace=True)
    df.to_csv(fpath[:-4]+'_casted.csv', index=False)


# In all the dataframe above the columbs will be: ['user_from', 'time', 'user_to']

def filter_dataframe_dates(df, date_range=None):
    '''
        This function filters the given dataframe leaving only the rows that represents the 
        interaction occourred in the given date_range

        Arguments
        ---------
            df: pd.Dataframe
                the dataframe to be filtered where the field time represent the timestamp of the interaction
            date_range: (str,str) | (int, int) | None
                if given represents the interval of date that are of interest for us
        Returns
        -------
            The filtered dataframe
    '''
    if date_range is None:
        return df
    dataframes = []
    for dr in date_range:
        if dr is not None and (type(dr) != tuple or len(dr)!=2):
            raise ValueError(f"You must pass a range of timestamp in the form: (min_ts, max_ts), not {date_range}")
        if dr is not None and type(dr[0]) == str:
            dr = tuple(map(lambda t: int(datetime.strptime(t, '%Y-%m-%d').timestamp()), dr))
        if dr is not None:
            dataframes.append(df[df.time.isin(range(*dr))])
    return pd.concat(dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)


def time_to_weight(df):
    '''
        This functions replace the content of the field time with the number of interaction between the users 
        in all the dataframe
        EX:
            user1 user2 2008-09-24
            user1 user2 2008-09-25
            user1 user2 2008-09-14
            user1 user3 2008-09-24
        
            became:
            user1 user2 3
            user1 user3 1

        Arguments
        ---------
            df: pd.Dataframe
                the dataframe to be processed 
        Returns
        -------
            The processed dataframe
    '''
    return df.groupby(['user_from', 'user_to'], as_index=False).agg({'time':'count'})


def get_graph_from_weighted_df(df_weighted):
    '''
        This function returns a graph in the form of adiacency list represented by a dictionary:
        {
            node1: [(node2, weight), (node3, weight),(nodek, weight)],
            node3: [(node2, weight)],
            noden: [(node5, weight), (node5, weight),(nodek, weight)]
        }

        Arguments
        ---------
            df: pd.Dataframe
                the dataframe from which take the nodes and the weights of the edges (time will represent the weights)
        Returns
        -------
            The dictionary of the obtained graph
    '''
    df_grouped_from = df_weighted.groupby(['user_from'])[['user_to', 'time']].apply(lambda pair: list(zip(pair['user_to'].values,pair['time'].values)))
    return df_grouped_from.to_dict()


def get_graph_from_df(df, date_range=None):
    '''
        This function returns a graph in the form of adiacency list represented by a dictionary:
        {
            node1: [(node2, weight), (node3, weight),(nodek, weight)],
            node3: [(node2, weight)],
            noden: [(node5, weight), (node5, weight),(nodek, weight)]
        }

        Arguments
        ---------
            df: pd.Dataframe
                the dataframe from which build the graph
            date_range: (str,str) | (int, int) | None
                if given represents the interval of date that are of interest for us   
        Returns
        -------
            The dictionary of the obtained graph
    '''
    if date_range is not None and type(date_range) != list:
        df = filter_dataframe_dates(df, [date_range])
    else:
        df = filter_dataframe_dates(df, date_range)
    
    return get_graph_from_weighted_df(time_to_weight(df))


def get_single_graph(fpath, date_range=None):
    '''
        This function returns a graph object of the class MyGraph

        Arguments
        ---------
            fpath: str
                the path where the dataframe describing the graph can be retrieved
            date_range: (str,str) | (int, int) | None
                if given represents the interval of date that are of interest for us   
        Returns
        -------
            The MyGraph object representing the graph
    '''

    start = time()
    print(f"Reading the file: {fpath}")
    df = pd.read_csv(fpath, header='infer')
    print(f"done in {round(time()-start,3)}s")
    start = time()
    print("Retrieving the graph...")
    if date_range is not None and type(date_range)!=list:
        ret = MyGraph(time_to_weight(filter_dataframe_dates(df, [date_range])))
    else:
        ret = MyGraph(time_to_weight(filter_dataframe_dates(df, date_range)))
    print(f"done in {round(time()-start,3)}s")
    return ret


def get_global_graph(fpaths, date_range=None, coefficients=None):
    '''
        This function returns a graph object containing all the nodes coming from the three files

        Arguments
        ---------
            fpaths: List[str]
                the paths where the dataframes describing the graphs can be retrieved in the form:
                [answer_to_questions, comment_to_answers, comment_to_questions]
            date_range: (str,str) | (int, int) | None
                if given represents the interval of date that are of interest for us   
            coefficiente: List[int]
                List of the coefficient with which we want to give importance to specific edge tipes:
                [a2q_coeff, c2a_coeff, c2q_coeff]
        Returns
        -------
            The MyGraph object representing the graph

        Warning
        -------
            Using big date_range is highly discouraged, the execution time can be really really high given 
            the huge amount of edges (more than 60M)
    '''
    
    if type(fpaths)!=list or len(fpaths)!=3:
        raise ValueError(f"the fpaths list must be the list containing in the order the [a2q, c2a, c2q] dataframe's paths, not {fpaths}")
    
    if coefficients is not None and (type(coefficients)!=list or len(coefficients)!=3):
        raise ValueError(f"The coefficients list must be a list containing three integer that represents the importance"+
                                +f" of the dataframe in the relative position at fpaths, not  {coefficients}")
    
    start = time()
    first_start = start
    print(f"reading the files:\n"+\
            f"\t-answer to questions: {fpaths[0]}"+\
            f"\t-comment to answers: {fpaths[1]}"+\
            f"\t-comment to questions: {fpaths[2]}")
    a2q = pd.read_csv(fpaths[0], header='infer')
    c2a = pd.read_csv(fpaths[1], header='infer')
    c2q = pd.read_csv(fpaths[2], header='infer')
    print(f"done in {round(time()-start, 3)}s")


    start = time()
    print(f"filtering the dataframes...")
    if date_range is not None and type(date_range)!=list:
        a2q = filter_dataframe_dates(a2q, [date_range])
        c2a = filter_dataframe_dates(c2a, [date_range])
        c2q = filter_dataframe_dates(c2q, [date_range])
    else:
        a2q = filter_dataframe_dates(a2q, date_range)
        c2a = filter_dataframe_dates(c2a, date_range)
        c2q = filter_dataframe_dates(c2q, date_range)
    print(f"done in {round(time()-start, 3)}s")

    start = time()
    print("generating the weights")
    a2q = time_to_weight(a2q)
    c2q = time_to_weight(c2q)
    c2a = time_to_weight(c2a)
    if coefficients is not None:
        print(f"- multiplying by the coefficients: {coefficients}")
        a2q.time=(a2q.time * coefficients[0])
        c2a.time=(c2q.time * coefficients[1])
        c2q.time=(c2a.time * coefficients[2])
    print(f"done in {round(time()-start, 3)}s")

    start = time()
    print("putting all togheter, it may require some time...")
    partial = pd.concat((a2q,c2q)).groupby(['user_from', 'user_to'], as_index=False).sum()
    tot = pd.concat((partial,c2a)).groupby(['user_from', 'user_to'], as_index=False).sum()
    print(f"done in {round(time()-start, 3)}s")

    start = time()
    print("retrieving the graph")
    ret = MyGraph(tot)
    print(f"done in {round(time()-start, 3)}s tot time elapsed: {round(time()-first_start, 3)}")
    return ret



class MyGraph():
    '''
        Class representing a graph.
        It's represented by the adiacency list in dictionary form.
    '''

    def __init__(self, df):
        '''
            Build the MyGraph instance starting from a dataframe with the weight in the time column
        '''
        self.adiacency_lists = get_graph_from_weighted_df(df)
        self.nodes = set(df.user_from.unique()).union(set(df.user_to.unique()))
        self.nx_graph = None
    
    def get_nx_graph(self):
        '''
            Returns the networkx graph built by this instance
        '''
        if self.nx_graph is None:
            ret = nx.DiGraph()
            for n1, l in tqdm(self.adiacency_lists.items(), total=len(self.adiacency_lists)):
                ret.add_weighted_edges_from((n1,b,w) for b,w in l)
            self.nx_graph = ret
        return self.nx_graph