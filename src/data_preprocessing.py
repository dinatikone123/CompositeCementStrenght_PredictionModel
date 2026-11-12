# Step 2 : Data preprocessing and cleaning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(df) :
  '''
  Step 1  : Feature engineering and Feature selection

  Step 2 : Split the data into independent and dependent columns i.e X and y.

  Step 3 : Split the data into seen and unseen data i.e train and test data.

  Step 4 : Scaling / Normalizing data(Standard scaler, min max scaler, robust scaler)

  Step 5 : Return the seen and unseen data from the function
  '''

  # Step 1 : Feature Enginerring and Feature selection
  df['Water_to_Cementitious_MaterialRatio'] = df['Water  (component 4)(kg in a m^3 mixture)'] / (df['Cement (component 1)(kg in a m^3 mixture)'] +
                                                                                                df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] +
                                                                                                df['Fly Ash (component 3)(kg in a m^3 mixture)'])

  # Step 2 : Split the data into independent and dependent columns i.e X and y.
  target_col = 'Concrete compressive strength(MPa, megapascals) '
  X = df.drop(columns = [target_col])
  y = df[target_col]

  # Step 3 : Split data in train and test
  X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state= 1)
  # Step 4 : Scaling / Normalizing data
  sc = MinMaxScaler()
  X_train = sc.fit_transform(X_train) # seen
  X_test = sc.transform(X_test)       # unseen

  return X_train,X_test,y_train,y_test