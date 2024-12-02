import kagglehub
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_dataset():
  path = kagglehub.dataset_download("Microsoft/microsoft-security-incident-prediction")
  print("Path to dataset files:", path)
  
  df_train = pd.read_csv(path+'/GUIDE_Train.csv',low_memory=False)
  df_test = pd.read_csv(path+'./GUIDE_Test.csv',low_memory=False)
  return df_train,df_test

def preprocess_data(df_train,df_test):
  features = ['DeviceId',
        'Sha256',
        'IpAddress',
        'Url',
        'NetworkMessageId',
        'RegistryKey',
        'RegistryValueData',
        'ApplicationName',
        'AccountUpn',
        'OSVersion',
        'Timestamp',
        'Category',
        'MitreTechniques',
        'ActionGrouped',
        'ActionGranular',
        'EntityType',
        'EvidenceRole',
        'EmailClusterId',
        'ThreatFamily',
        'ResourceType',
        'Roles',
        'AntispamDirection',
        'SuspicionLevel',
        'LastVerdict']
    
    df_train = df_train.dropna(subset=['IncidentGrade'])
    df_test = df_test.dropna(subset=['IncidentGrade'])

    df_train = df_train.fillna('')
    df_test = df_test.fillna('')


    for col_name in features:
        df_train[col_name] = df_train[col_name].astype(str)

    for col_name in features:
        df_test[col_name] = df_test[col_name].astype(str)

    X_train = df_train[features]
    y_train = df_train['IncidentGrade']
    X_test = df_test[features]
    y_test = df_test['IncidentGrade']

return X_train,y_train,X_test,y_test,features

def train_model(X_train,y_train,X_test,y_test,features):
    # Create CatBoost Pool objects
    train_pool = Pool(data=X_train, label=y_train, cat_features=features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=features)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=1000,  # Adjust as needed
                            depth=7,          # Adjust as needed
                            learning_rate=0.1, # Adjust as needed
                            loss_function='MultiClass',  # For multi-class classification
                            random_seed=42,
                            # l2_leaf_reg= 5,
                            verbose=100)     # Adjust for verbosity

    # Train the model
    model.fit(train_pool)

    #save model
    model.save_model("model.json", format="json")
    return test_pool

def test_model(test_pool):
    #load model
    loaded_model = CatBoostClassifier()
    loaded_model.load_model("model.json", format="json")
    # Make predictions
    predictions = loaded_model.predict(test_pool)
  
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted')}")
    print(f"Recall: {recall_score(y_test, predictions, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")


def main():
    df_train,df_test = load_dataset()
    X_train,y_train,X_test,y_test,features = preprocess_data(df_train,df_test)
    test_pool = train_model(X_train,y_train,X_test,y_test,features)
    test_model(test_pool)
    
if __name__ == "__main__":
    main()
