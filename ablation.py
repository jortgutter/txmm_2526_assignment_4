from model import Model

def ablation_test(
    model_class:Model, 
    model_args:dict,
    X_train, 
    X_test, 
    y_train, 
    y_test,
    feature_dict,
    ax, 
    is_last, 
):
    all_feature_names = feature_dict['all_features']
    scores = {}
    colors = []
    alphas = []
    # get the base score when using all features:
    scores['None'] = model_class.train_and_score(
        model_args=model_args,
        X_train=X_train.copy(deep=True),
        y_train=y_train.copy(deep=True),
        X_test=X_test.copy(deep=True),
        y_test=y_test.copy(deep=True),
        feature_names=all_feature_names
    )
    colors.append('C0')
    alphas.append(1)
    # perform the actual ablation tests:
    feature_classes = [feature_class for feature_class in feature_dict.keys() if type(feature_class) != str]
    for i, feature_class in enumerate(feature_classes):
        
        prettified_feature_group_name = f'all {feature_class.group_name}'
        group_features = feature_dict[feature_class]['feature_names']
        
        # remove all group features
        ablated_feature_names = all_feature_names.copy() 
        for feature in group_features:
            ablated_feature_names.remove(feature)  
        
        # train and test the model
        
        scores[prettified_feature_group_name] = model_class.train_and_score(
            model_args=model_args,
            X_train=X_train.copy(deep=True),
            y_train=y_train.copy(deep=True),
            X_test=X_test.copy(deep=True),
            y_test=y_test.copy(deep=True),
            feature_names=ablated_feature_names
        )
        colors.append(f'C{i+1}')
        alphas.append(1)
        
        if feature_dict[feature_class]['ablate_individual_features']:
        # remove one by one
            for feature in group_features:
        
                # copy the feature list and remove one:
                ablated_feature_names = all_feature_names.copy() 
                ablated_feature_names.remove(feature)  
                
                # for nicer printing in the plot func:
                prettified_feature_name = feature.replace("_", " ")  
                
                # train and test the model
                scores[prettified_feature_name] = model_class.train_and_score(
                    model_args=model_args,
                    X_train=X_train.copy(deep=True),
                    y_train=y_train.copy(deep=True),
                    X_test=X_test.copy(deep=True),
                    y_test=y_test.copy(deep=True),
                    feature_names=ablated_feature_names
                )
                colors.append(f'C{i+1}')
                alphas.append(0.5)
        
    bar_labels = []
    score_keys = []
    score_values = []
    for key, val in scores.items():
        score_keys.append(key)
        score_values.append(val)
    bars = ax.bar(score_keys, score_values)
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        bar.set_alpha(alphas[i])
       
        if alphas[i] == 1: 
            bar_labels.append(f'{score_values[i]:.2f}')
        else:
            bar_labels.append('')
        
    if is_last:
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_rotation(45)
            label.set_ha('right')
            if alphas[i] == 1:
                label.set_fontweight('bold')
        ax.set_xlabel('Ablated feature')
    else:
        ax.set_xticks([])
        
    ax.bar_label(bars, labels=bar_labels, padding=3)
    ax.set_ylabel('F1-score')
    ax.set_ylim(0, 1)
    ax.set_title(f'{model_class.__name__}')