import pandas as pd
import typing

class FeatureExtractor:
    group_name = 'Default Extractor -- THIS SHOUD NEVER BE VISIBLE'
    @classmethod
    def get_group_name(cls):
        return cls.group_name
    
    @staticmethod
    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
        print(f'This should NEVER BE PRINTED')
        return None, None, None