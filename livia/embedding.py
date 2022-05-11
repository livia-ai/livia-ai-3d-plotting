import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class Embedding():
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def generate_embedding(self, dataframe:pd.DataFrame, emb_column_names:list, id_column_name:str, embedding_dimensions:int=None):
        """
        :dataframe: loaded data as pandas DataFrame
        :emb_column_names: list of column  namess that should be considered for the embedding
        :id_column_name: name of sample identifier column

        returns: embeddings
        """

        # add id column to the front of the columns list
        emb_column_names.insert(0, id_column_name)

        print("-> Dataframe is processed: Columns are merged; Text data is preprocessed; ...")
        # filter dataframe to contain only id_column + emb_columns
        df_filtered = dataframe[emb_column_names]

        # create column full text that contains all the text from emb_columns
        df_filtered = df_filtered.assign(full_text = df_filtered[df_filtered.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1))

        # apply small text preprocessing pipeline
        df_filtered = self.preprocessing(df_filtered, "full_text")

        # create list of strings -> input format for model
        text_list = list()
        for text in df_filtered["pre_text"]:
            text_list.append(text)
        
        # generate sentence embedding
        print("-> Sentence embedding is generated:")
        sentence_embeddings = self.model.encode(text_list, show_progress_bar=True, device="cpu")
        
        # if necessary downproject the computed embeddings
        if embedding_dimensions != None:
            # project to d dimensions
            stand_sentence_embeddings = (sentence_embeddings - np.mean(sentence_embeddings, axis=0)) / np.std(sentence_embeddings, axis=0)
            pca = PCA(n_components=embedding_dimensions)
            sentence_embeddings = pca.fit_transform(stand_sentence_embeddings)


        # concatenate the index column in the first column of the array
        identifier = df_filtered[id_column_name].to_numpy().reshape((-1,1))
        sentence_embeddings = np.concatenate([identifier, sentence_embeddings], axis=1)
        # small format hack incase identifier is a str as in belvedere dataset
        format = ['%s']
        format += ['%.18e']*(sentence_embeddings.shape[1]-1)
        # save down projected sentence embedding
        np.savetxt(f'sentence_embedding_{sentence_embeddings.shape[1]-1}d.csv', sentence_embeddings, delimiter=',', fmt=format)

    def preprocessing(self, text_data: pd.DataFrame, column_name: str) -> pd.DataFrame:

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
