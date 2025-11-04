from feature_extractor import FeatureExtractor
import pandas as pd
import typing
from sklearn.feature_extraction.text import TfidfVectorizer

class BOWFeatures(FeatureExtractor):
    group_name = 'BOW Features'
    
    @staticmethod
    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
        vectorizer=feature_parameters
        if vectorizer is None:
            #normalizing the score and extracting only the top 10 words
            vectorizer = TfidfVectorizer(max_features=10)
            matrix = vectorizer.fit_transform(data_frame['text'])
        else:
            matrix = vectorizer.transform(data_frame['text'])
        feature_names = [
            f'Tfidf_{w}' for w in vectorizer.get_feature_names_out()
        ]
        bow_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=data_frame.index)
        data_frame = pd.concat([data_frame, bow_df], axis=1)
        return data_frame, feature_names, vectorizer
