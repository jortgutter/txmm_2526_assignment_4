from feature_extractor import FeatureExtractor
import pandas as pd
import typing

class POSFeatures(FeatureExtractor):
    group_name = 'POS Features'
    
    @staticmethod
    def tag_counter(pos_tagged_sentences, tag):
        tag_sum = 0
        for sentence in pos_tagged_sentences:
            for token_tag_pair in sentence:
                if token_tag_pair[1] == tag:
                    tag_sum += 1
        return tag_sum
    
    @staticmethod
    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
        chosen_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 
    'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
    'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
    'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        
        
        assert 'pos_tagged_sentences' in data_frame.columns
        
        feature_names = []
        
        for tag in chosen_tags:
            feature_name = f'pf_{tag}'
        
            data_frame[feature_name] = data_frame['pos_tagged_sentences'].apply(
                lambda x: POSFeatures.tag_counter(x, tag)
            ) / data_frame['token_count']

        tag_sums = {}
        for tag in chosen_tags:
            feature_name = f'pf_{tag}'
            tag_sums[tag] = data_frame[feature_name].sum()
        
        top_tags = sorted(tag_sums, key=tag_sums.get, reverse=True)[:10]

        for tag in top_tags:
            feature_name = f'pf_{tag}'
            feature_names.append(feature_name)
        
        return data_frame, feature_names, None