from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold

def get_transformer():
    return FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy='mean')),
            ('indicators', MissingIndicator())]
    )

def get_pipeline(model):
    return make_pipeline(get_transformer(), model)

def fit(clf, params, cv=10, X_train=X_train, y_train=y_train):
    grid = GridSearchCV(
        clf, 
        params, 
        cv=KFold(n_splits=cv), 
        n_jobs=-1, 
        verbose=1, 
        return_train_score=True, 
        scoring='accuracy', 
        refit=True
    )
    grid.fit(X_train, y_train)
    return grid

def best_scores(model):
    # print(f'The mean cross validation test score is: {model.cv_results_.mean_test_score}') #for some reason this wasn't working for me even though the attribute exists so lets just leave it.
    print(f'The best parameters are: {model.best_params_}')
    print(f'The best score that we got is: {model.best_score_}')
    return None

def check_scores(y_test, y_pred):
    # print('Precision: %.3f' % precision_score(y_test, y_pred))
    # print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
    # print('F1 Score: %.3f' % f1_score(y_test, y_pred))
    # print('ROC-AUC Score: %.3f' % roc_auc_score(y_test, y_pred))
