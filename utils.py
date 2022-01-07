from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import FeatureUnion, make_pipeline


def get_transformer():
    return FeatureUnion(
        transformer_list=[
            ("features", SimpleImputer(strategy="mean")),
            ("indicators", MissingIndicator()),
        ]
    )


def get_pipeline(model):
    return make_pipeline(get_transformer(), model)


def fit(clf, params, X_train, y_train, cv=10):
    grid = GridSearchCV(
        clf,
        params,
        cv=KFold(n_splits=cv),
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        scoring="accuracy",
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def best_scores(model):
    print(f"The best parameters are: {model.best_params_}")
    print(f"The best score that we got is: {model.best_score_}")
    print(f"The best estimator is: {model.best_estimator_}")
    
    return None


def check_scores(y_test, y_pred):
    print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))


def test_model(model, params, X, y, cv=10):
    model = get_pipeline(model)
    model = fit(model, params, X, y, cv)
    best_scores(model)
    return model
