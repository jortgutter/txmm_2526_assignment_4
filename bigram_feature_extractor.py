from feature_extractor import FeatureExtractor
import pandas as pd
import typing

from sklearn.feature_extraction.text import TfidfVectorizer


class BigramFeatures(FeatureExtractor):
    group_name = 'Bigram Features'
    
    def get_features(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
        vectorizer = feature_parameters

        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(2, 2),
                max_features=500,           # top n bigrams
                stop_words='english'      # auto filter stopwords
            )
            matrix = vectorizer.fit_transform(data_frame['text'])
        else:
            matrix = vectorizer.transform(data_frame['text'])
            
        feature_names = [f'bigram_{name}' for name in vectorizer.get_feature_names_out()]
        # add columns to dataframe
        bigram_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=data_frame.index)
        data_frame = pd.concat([data_frame, bigram_df], axis=1)
        
        return data_frame, feature_names, vectorizer
        
            
            
            
            
            
            
    # def get_features_old(data_frame: pd.DataFrame, feature_parameters:typing.Any=None) -> tuple[pd.DataFrame, list[str], typing.Any]:
    #     n_bigrams = feature_parameters
    #     topn_list = []
    #     for tokenized_text in data_frame['tokens']:
    #         bigram_list = list(bigrams(tokenized_text))
    #         bigram_stopwords_removed_list = BigramFeatures.remove_stopwords_pairs(bigram_list)
        
    #         if not bigram_stopwords_removed_list:
    #             topn_list.append({})
    #             continue

    #         freq_dist = FreqDist(bigram_stopwords_removed_list)
    #         prob_dist = MLEProbDist(freq_dist)

    #         bigram_prob_dict = {bg: prob_dist.prob(bg) for bg in freq_dist.keys()}

    #         sorted_dict = dict(sorted(bigram_prob_dict.items(), key=lambda item: item[1], reverse=True)[:n_bigrams])

    #         topn_list.append(sorted_dict)

        
    #     bigram_prob_df = pd.DataFrame(topn_list).fillna(0)
    #     bigram_prob_df.columns = [f'bigram_{bg}' for bg in bigram_prob_df.columns]

    #     data_frame = pd.concat([data_frame.reset_index(drop=True), bigram_prob_df.reset_index(drop=True)], axis=1)
    #     feature_names = list(bigram_prob_df.columns)
    #     return data_frame, feature_names



    # #removing pairs consisting only stopwords
    # def remove_stopwords_pairs(ngram):
    #     stop_words = set(stopwords.words('english'))
    #     removed = []
    #     for pair in ngram:
    #         count = 0
    #         for word in pair:
    #             if word not in stop_words:
    #                 count = 1
    #                 break
    #         if count == 0:
    #             removed.append(pair)
    #     return removed