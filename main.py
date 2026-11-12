# Orchestron Layer design : Entry point modeling
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_build import model_build
def main() : # Function Definition

# Step 1 : Data loading
  df = data_ingestion()

# Step 2 : Data Preprocessing and cleaning
  X_train,X_test,y_train,y_test = data_preprocessing(df)

#step3 : Model Building
  scores = model_build(X_train,X_test,y_train,y_test)
  print(f'R2 score: {scores}')

main()       # Function calling
