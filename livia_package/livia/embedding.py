import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class Embedding:
    def __init__(self, embedding, identifier):
        self.embedding = embedding
        self.shape = self.embedding.shape
        self.identifier = identifier

def save_to_csv(embedding:Embedding, file_name:str):
    format = ['%s']
    format += ['%.18e']*(embedding.shape[1])
    np.savetxt(file_name+".csv",np.concatenate( [embedding.identifier, embedding.embedding], axis=1), delimiter=',', fmt=format)

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
    embedding = model.encode(text_list, show_progress_bar=True, device="cpu")

    # create embedding object containing vectors and identifiers
    identifier = df_filtered[id_column_name].to_numpy().reshape((-1,1))

    embedding_object = Embedding(embedding, identifier)
    return embedding_object

def pca_reduce(embedding:Embedding, dimensions:int):

    # extract computed sentence embeddings
    matrix = embedding.embedding
    # standardize the stence embeddings
    stand_embedding = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    # perform pca
    pca = PCA(n_components=dimensions)
    downprojected_embedding = pca.fit_transform(stand_embedding)

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
