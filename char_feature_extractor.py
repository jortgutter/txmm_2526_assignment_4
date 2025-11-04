from feature_extractor import FeatureExtractor
import pandas as pd
import typing

class CharFeatures(FeatureExtractor):
    group_name = 'Char Features'

    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:

        # some char count fetaures
        
        data_frame['cf_comma'] = data_frame['text'].str.count(r',') / data_frame['snippet_length']
        data_frame['cf_period'] = data_frame['text'].str.count(r'\.') / data_frame['snippet_length']
        data_frame['cf_exclam'] = data_frame['text'].str.count(r'\!') / data_frame['snippet_length']
        data_frame['cf_question'] = data_frame['text'].str.count(r'\?') / data_frame['snippet_length']
        data_frame['cf_upper_case'] = data_frame['text'].str.count(r'[A-Z]') / data_frame['snippet_length']
        data_frame['cf_lower_case'] = data_frame['text'].str.count(r'[a-z]') / data_frame['snippet_length']
        
        # some aggregated char count features
        data_frame['cf_vowel_frequency'] = data_frame['text'].str.count(r'[aeiou]') / data_frame['snippet_length']
        data_frame['cf_avg_word_len'] = data_frame['text'].apply(lambda x: sum(len(w) for w in x.split()) / len(x.split()) if x.split() else 0)
        
        # make sure any features you want to be used during training are also in this list:
        feature_names = [
            'cf_comma', 
            'cf_period', 
            'cf_exclam', 
            'cf_question', 
            'cf_upper_case',
            'cf_lower_case',
            'cf_vowel_frequency',
            'cf_avg_word_len'
        ]
        
        return data_frame, feature_names, None