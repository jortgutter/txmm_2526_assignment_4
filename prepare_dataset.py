import nltk
from nltk import sent_tokenize
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPrep:
    def pos_tagger(sentences):
        return [nltk.pos_tag(word_tokenize(sent)) for sent in sentences]

    def prepare(data_frame):
        data_frame['snippet_length'] = data_frame['text'].str.len()
        
        data_frame['tokens'] = data_frame['text'].apply(
            lambda x: word_tokenize(x.lower())
        )
        data_frame['token_count'] = data_frame['tokens'].apply(
            lambda x: len(x)
        )
        data_frame['sentences'] = data_frame['text'].apply(
            lambda x: sent_tokenize(x.lower())
        )
        data_frame['sentence_count'] = data_frame['sentences'].apply(
            lambda x: len(x)
        )
        data_frame['pos_tagged_sentences'] = data_frame['sentences'].apply(
            lambda x: DataPrep.pos_tagger(x)
        )
        return data_frame
    
    def extract_features(
        feature_dict, 
        X_train, 
        X_test
    ):
        """extracts relevant features"""
        # prepare trainset and testset
        print(f'Extracting features for X_train: ...')
        train_data_frame_prepared = DataPrep.prepare(X_train)
        print(f'Extracting features for X_test: ...')
        test_data_frame_prepared = DataPrep.prepare(X_test)
        
        # list of feature names we want to use for training
        feature_names = []
        
        for feature_class in feature_dict.keys():
            
            # extract features for the train set:
            train_data_frame_prepared, ft_names, feature_parameters = feature_class.get_features(
                data_frame=train_data_frame_prepared,
                feature_parameters=None
            )
            # extract features for the test set:
            test_data_frame_prepared, _, _ = feature_class.get_features(
                data_frame=test_data_frame_prepared, 
                feature_parameters=feature_parameters
            )
            # add feature names to the feature names list:
            feature_names += ft_names
            # add feature group to feature groups dict:
            feature_dict[feature_class]['feature_names'] = ft_names
        
        feature_dict['all_features'] = feature_names    
        # return the data with the features and the list of feature names
        return train_data_frame_prepared, test_data_frame_prepared, feature_dict
    
