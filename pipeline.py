from sklearn.model_selection import train_test_split
from prepare_dataset import DataPrep
import matplotlib.pyplot as plt
from ablation import ablation_test

def train_and_test_all(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_dict,
    model_dict
):# dictionary with the selected model types and their optional arguments:

    n_models=len(model_dict)
    fig, axs = plt.subplots(n_models,1, figsize=(12,5*n_models))
    
    print(f'Training and testing {n_models} models')
    # perform an ablation test on each type of model
    for i, model_class in enumerate(model_dict.keys()):
        ablation_test(
            model_class=model_class,
            model_args=model_dict[model_class],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_dict=feature_dict,
            ax=axs[i],
            is_last=i==n_models-1,
        )
    plt.suptitle('Ablation Results')
    plt.tight_layout()
    plt.show()


def evaluate_on_dev(dev_set, feature_dict, model_dict):
    X_df = dev_set
    y_df = dev_set['author']
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    
    X_train, X_test, feature_dict = DataPrep.extract_features(
        feature_dict=feature_dict, 
        X_train=X_train,
        X_test=X_test
    )
    
    train_and_test_all(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_dict=feature_dict,
        model_dict=model_dict
    )
    
def evaluate_on_train_test(train_set, test_set, feature_dict, model_dict):
    X_train = train_set
    X_test = test_set
    y_train = train_set['author']
    y_test = test_set['author']
    
    X_train, X_test, feature_dict = DataPrep.extract_features(
        feature_dict=feature_dict, 
        X_train=X_train,  
        X_test=X_test
    )

    train_and_test_all(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_dict=feature_dict,
        model_dict=model_dict
    )