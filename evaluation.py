from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier()
model.load_model("model.cbm") 

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


df_test = pd.read_csv('./GUIDE_Test.csv',low_memory=False)

df_test = df_test.dropna(subset=['IncidentGrade'])
df_test = df_test.fillna('')

for col_name in features:
        df_test[col_name] = df_test[col_name].astype(str)

    
X_test = df_test[features]
y_test = df_test['IncidentGrade']


test_pool = Pool(data=X_test, label=y_test, cat_features=features)

# Make predictions
predictions = model.predict(test_pool)


# Evaluate the model (example using accuracy)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision_score(y_test, predictions, average='weighted')}")
print(f"Recall: {recall_score(y_test, predictions, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
