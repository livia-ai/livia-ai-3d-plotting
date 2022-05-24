from re import X
from turtle import color
import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm

class Embedding:
    def __init__(self, embedding, identifier):
        self.embedding = embedding
        self.shape = self.embedding.shape
        self.identifier = identifier

def save_to_csv(embedding:Embedding, file_name:str):
    format = ['%s']
    format += ['%.18e']*(embedding.shape[1])
    np.savetxt(file_name+".csv",np.concatenate([embedding.identifier.reshape(-1,1), embedding.embedding], axis=1, dtype=object), delimiter=',', fmt=format)

def load_csv(path:str):
    identifier = np.loadtxt(path, delimiter=',', usecols=0, dtype=str)
    embedding = np.loadtxt(path, delimiter=',', dtype=object)[:, 1:].astype(np.float64)
    return Embedding(embedding, identifier)
    
def compute_embedding(dataframe:pd.DataFrame, emb_column_names:list, id_column_name:str) -> Embedding:
    """
    :dataframe: loaded data as pandas DataFrame
    :emb_column_names: list of column  namess that should be considered for the embedding
    :id_column_name: name of sample identifier column

    returns: instance of class Embedding containing the computed vectors and the identifier
    """

    # add id column to the front of the columns list
    emb_column_names.insert(0, id_column_name)

    print("-> Dataframe is processed: Columns are merged; Text data is preprocessed; ...")
    # filter dataframe to contain only id_column + emb_columns
    df_filtered = dataframe[emb_column_names]

    # create column full text that contains all the text from emb_columns
    df_filtered = df_filtered.assign(full_text = df_filtered[df_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))

    # apply small text preprocessing pipeline
    df_filtered = preprocessing(df_filtered, "full_text")

    # create list of strings -> input format for model
    text_list = list()
    for text in df_filtered["pre_text"]:
        text_list.append(text)
    
    #load model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # generate sentence embedding
    print("-> Sentence embedding is generated:")
    embedding = model.encode(text_list, show_progress_bar=True)

    # create embedding object containing vectors and identifiers
    identifier = df_filtered[id_column_name].to_numpy().reshape((-1,1))

    embedding_object = Embedding(embedding, identifier)
    return embedding_object

def pca_reduce(embedding:Embedding, dimensions:int, standardize=True):

    # extract computed sentence embeddings
    stand_embedding = embedding.embedding
    if standardize:
        # standardize the stence embeddings
        stand_embedding = (stand_embedding - np.mean(stand_embedding, axis=0)) / np.std(stand_embedding, axis=0)
    # perform pca
    pca = PCA(n_components=dimensions)
    downprojected_embedding = pca.fit_transform(stand_embedding).astype(np.float64)

    # extract identifier
    identifier = embedding.identifier

    return Embedding(downprojected_embedding, identifier)

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
        
        helper.at[i,"pre_text"] = ' '.join(text)
        i += 1
        
    return helper

def plot_3d(embedding_to_plot:Embedding, meta_data:pd.DataFrame, n:int, id_column, title_column, color_column, info_columns:list, title_plot="3D Plot of Embedding"):


    if embedding_to_plot.shape[1] > 3:
        embedding_to_plot = pca_reduce(embedding_to_plot,3)

    # make column list is unique
    column_list = [id_column, title_column, color_column] + info_columns
    columns_unique = list(dict.fromkeys(column_list))
    
    df = meta_data[columns_unique]
    df = df.astype({id_column: "str"})
    df = df.copy()

    # just in case order the 
    order_of_embedding = np.where(embedding_to_plot.identifier == df[id_column])
    embedding_matrix_3d = embedding_to_plot.embedding[order_of_embedding]


    embedding_length = len(embedding_matrix_3d)
    # if specified take random subsample of dataset
    if n != None:
        if n < embedding_length:
            rng = np.random.default_rng()
            id_list = list(range(embedding_length))
            sample_ids = rng.choice(id_list, size=n, replace=False)
            embedding_matrix_3d = embedding_matrix_3d[sample_ids]
            df = df.copy().loc[sample_ids]

    # for better visualization crop title
    length = 75
    df[title_column] = df[title_column].apply(lambda x: str(x)[:length] if len(str(x))>length else str(x))
    df[color_column] = df[color_column].apply(lambda x: str(x)[:100] if len(str(x))>100 else str(x))
    df["x"] = embedding_matrix_3d[:,0]
    df["y"] = embedding_matrix_3d[:,1]
    df["z"] = embedding_matrix_3d[:,2]
    df.fillna('NaN', inplace=True)

    fig = px.scatter_3d(df, 
                    x='x', y='y', z='z', 
                    color=color_column, 
                    hover_name=title_column, # what to show when hovered over
                    hover_data=[id_column] + info_columns,
                    width=2000, height=850, # adjust height and width
                    title=title_plot)
    
    # make set size for legend and hover label
    fig.update_layout(showlegend=True,
                     legend = dict(
                            font = dict(size = 10)
                            ), 
                    hoverlabel=dict(
                            font_size=10,
                            ),
                    title_x=0.5
                    )

    # set marker size
    fig.update_traces(marker_size = 3)
    fig.show()

