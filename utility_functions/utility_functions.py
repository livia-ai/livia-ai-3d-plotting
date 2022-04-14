from time import time
import pandas as pd
import numpy as np
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords

import plotly.express as px
import altair as alt
import altair_saver as asav

def preprocessing(text_data: pd.DataFrame, column_name: str) -> pd.DataFrame:

    helper = text_data.copy(deep = True)
    helper = helper.assign(pre_text = helper[column_name]) 
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    cachedStopWords = stopwords.words('german')
    
    # noch kein Stemmer -> könnt ma auch probieren hab online ein bisschen geschaut und die
    # Qualität von deutschen Stemmern ist relativ bescheiden
    
    # iterate over all documents and tokenize each text
    i = 0
    for text in helper[column_name]:
        # text = text.lower()
        # remove special characters and do tokenization
        text = np.array(tokenizer.tokenize(text))
        #remove stopwords
        text = [word for word in text if not word in cachedStopWords]
        
        helper.at[i,"pre_text"] = text
        i += 1
        
    return helper

def plot(meta_data, embeddings, nr_samples, sent_vec_gen_method, dim_red_method, color):

    df = meta_data[:nr_samples][["classifications", "subjects"]]
    # title gets cut off after "length" characters, otherwise hoverlabel is too long
    length = 75
    df["title"] = meta_data["title"][:nr_samples].apply(lambda x: x[:length] if len(x)>length else x)
    df["x"] = embeddings[:,0]
    df["y"] = embeddings[:,1]
    df["z"] = embeddings[:,2]
    df.fillna('NaN', inplace=True)

    title = f"Visualization of: {sent_vec_gen_method} + {dim_red_method} + {nr_samples} Samples + Color:{color}"

    fig = px.scatter_3d(df, 
                        x='x', y='y', z='z', 
                        color=color, 
                        hover_name="title", # what to show when hovered over
                        width=2500, height=1250, # adjust height and width
                        title=title)

    # make set size for legend and hover label
    fig.update_layout(showlegend=True,
                     legend = dict(
                            font = dict(size = 10)
                            ), 
                    hoverlabel=dict(
                            font_size=9,
                            )
                    )

    # set marker size
    fig.update_traces(marker_size = 3)
    fig.show()

def plot_clustering(df, sent_vec_gen_method, dim_red_method, cluster_algo):

    title = f"Visualization of: {sent_vec_gen_method} + {dim_red_method} + {cluster_algo}"

    fig = px.scatter_3d(df, 
                        x="x", y="y", z="z", 
                        color="label", 
                        hover_name="label",
                        hover_data=["id", "title", "classifications"], # what to show when hovered over
                        width=2500, height=1250, # adjust height and width
                        title=title)

    # make set size for legend and hover label
    fig.update_layout(showlegend=True,
                     legend = dict(
                            font = dict(size = 10)
                            ), 
                    hoverlabel=dict(
                            font_size=9,
                            )
                    )
                    
    # set marker size
    fig.update_traces(marker_size = 3)
    fig.show()

def get_counts(dataframe, column_to_count, nr_clusters):

    counts = dataframe[column_to_count].value_counts().to_frame(name = "counts_full")
    counts.index.name = column_to_count
    counts.reset_index(inplace=True)

    for i in range(nr_clusters):
        only_one_cluster = dataframe.loc[dataframe['label'] == i]
        counts_cluster = only_one_cluster[column_to_count].value_counts().to_frame(name = f"counts_{i}")
        counts_cluster.index.name = column_to_count
        counts_cluster.reset_index(inplace=True)
        counts = pd.merge(counts, counts_cluster, on=column_to_count, how='left')
    counts = counts.fillna(0)

    return counts

def plot_counts_full(counts):
    alt.data_transformers.disable_max_rows()
    #alt.renderers.enable('altair_viewer')

    bars = alt.Chart(counts).mark_bar().encode(
        x=alt.X('classifications', sort='-y'),
        y='counts_full',
    )
    bars.configure_axis(labelLimit=400)

    asav.save(bars, f'bar_full.pdf')

def plot_counts_clusters(counts, nr_clusters, path):
    alt.data_transformers.disable_max_rows()
    alt.renderers.enable('altair_viewer')
    columns = [f'counts_{x}' for x in range(nr_clusters)]
    bars = alt.Chart(counts).transform_fold(
        columns
    ).mark_bar().encode(
        alt.X('classifications', sort='-y', axis=alt.Axis(labelAngle=-90)),
        y='value:Q',
        color=alt.Color('key:N', legend=alt.Legend(
            orient='none',
            legendX=130, legendY=-40,
            direction='horizontal',
            titleAnchor='middle'))
    ).properties(
        height=500
    )

    bars.configure_axis(labelLimit=400)
    
    asav.save(bars, f'{path}.pdf')

def time_function(function, x):
    start = time()
    output = function(**x)
    end = time()
    elapsed_time = end - start

    return output, elapsed_time