# Step 3 : Model Training and Evaluation

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,SGDRegressor,HuberRegressor,TheilSenRegressor,RANSACRegressor

from sklearn.metrics import r2_score

scores = []

def model_build(X_train,X_test,y_train,y_test) :
  models = {'LinerRegression' : LinearRegression(),
            'Ridge' : Ridge(),
            'Lasso' : Lasso(),
            'ElasticNet' : ElasticNet(),
            'SGDRegressor' : SGDRegressor(),
            'HuberRegressor': HuberRegressor(),
            'TheilSenRegressor' : TheilSenRegressor(),
            'RANSACRegressor' : RANSACRegressor()
            }

  for models_name, model in models.items():
    model.fit(X_train,y_train)        # Seen Data
    y_pred = model.predict(X_test)    # Unseen Data
    r2 = r2_score(y_test,y_pred)      # r2score = (1 - error)
    scores.append(r2)
    print(f'{models_name} R2 Score: {r2}')
  return scores



