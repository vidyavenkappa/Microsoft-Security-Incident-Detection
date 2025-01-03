import pandas as pd
from catboost import CatBoostClassifier, Pool


def main():
    df_train = pd.read_csv('./GUIDE_Train.csv',low_memory=False)

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

    df_train = df_train.fillna('')

    for col_name in features:
        df_train[col_name] = df_train[col_name].astype(str)

   
    X_train = df_train[features]
    y_train = df_train['IncidentGrade']
   
    
    # Create CatBoost Pool objects
    train_pool = Pool(data=X_train, label=y_train, cat_features=features)
    

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
    model.save_model("model.cbm") 


if __name__ == "__main__":
    main()
