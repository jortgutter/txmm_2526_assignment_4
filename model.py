from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

class Model:
    """
    A wrapper class that is meant to wrap around an arbitrary machine learning model
    """
    def __init__(self, model_args, X_train, y_train, feature_names):
        self.model_args=model_args
        self.X_train=X_train[feature_names]
        self.y_train=y_train
        self.feature_names=feature_names
    
    def fit(self):
        raise NotImplementedError

    def _test(self):
        raise NotImplementedError
    
    def _score_fn(self):
        raise NotImplementedError
        
    def predict(self, X_test, y_test):
        self.X_test=X_test[self.feature_names]
        self.y_test=y_test
        return self._test()
    
    def score(self, X_test, y_test):
        self.X_test=X_test[self.feature_names]
        self.y_test=y_test
        return self._score_fn()
    
    @classmethod
    def train_and_score(
        model_class, 
        model_args, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        feature_names
    ):
        model = model_class(model_args, X_train, y_train, feature_names)
        model.fit()
        return model.score(X_test, y_test)
    
class DecisionTree(Model):
    """
    Wrapper for the sklearn DecisionTree
    """
    def fit(self):
        self.model = DecisionTreeClassifier(**self.model_args)
        self.model.fit(self.X_train, self.y_train)
        
    def _test(self):
        # Make predictions
        return self.model.predict(self.X_test)
    
    def _score_fn(self):
        y_pred = self.model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')
    
class RandomForest(Model):
    """
    Wrapper for the sklearn RandomForest
    """
    def fit(self):
        self.model = RandomForestClassifier(**self.model_args)
        self.model.fit(self.X_train, self.y_train)
        
    def _test(self):
        # Make predictions for the test set
        return self.model.predict(self.X_test)
    
    def _score_fn(self):
        # Get the accuracy on the test set
        y_pred = self.model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')
    
class CatBoost(Model):
    def fit(self):
        self.model = CatBoostClassifier(**self.model_args)
        self.model.fit(self.X_train, self.y_train)
        
    def _test(self):
        # Make predictions for the test set
        return self.model.predict(self.X_test)
    
    def _score_fn(self):
        # Get the accuracy on the test set
        y_pred = self.model.predict(self.X_test)
        return f1_score(self.y_test, y_pred, average='weighted')
        
    