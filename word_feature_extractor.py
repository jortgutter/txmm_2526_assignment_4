from feature_extractor import FeatureExtractor
import pandas as pd
import typing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class WordFeatures(FeatureExtractor):
    group_name = 'Word Features'

    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
        # count number of words
        data_frame['wf_word_count'] = data_frame['text'].apply(
            lambda x: len(x.split())
        )
        # count the stopwords
        stop_words = set(stopwords.words('english'))
        data_frame['wf_stopword_frequency'] = data_frame['text'].apply(
            lambda x: sum(1 for w in word_tokenize(x.lower()) if w in stop_words)
        ) / data_frame['wf_word_count']
        
        # make sure any features you want to be used during training are also in this list:
        feature_names = [
            'wf_word_count', 
            'wf_stopword_frequency'
        ]
        
        return data_frame, feature_names, None