import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import plotly.express as px

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
    df["title"] = meta_data["id"][:nr_samples] #.apply(lambda x: x[:length] if len(x)>length else x)
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

