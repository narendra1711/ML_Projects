#importing libraries
import pandas as pd

#importing the dataset
df_building_ownership = pd.read_csv('./Dataset/Building_Ownership_Use.csv')
df_building_structure = pd.read_csv('./Dataset/Building_Structure.csv')
df_train = pd.read_csv('./Dataset/train.csv')
df_test = pd.read_csv('./Dataset/test.csv')

result = pd.merge(df_building_structure, df_building_ownership, on='building_id')
res_train=pd.merge(df_train,result,on="building_id")
res_test=pd.merge(df_test,result,on="building_id")

X_train = res_train.iloc[:, res_train.columns != 'damage_grade']
y_train = res_train.iloc[:, 2]
X_test = res_test.iloc[:, :]

#take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis=0)
imputer = imputer.fit(X_train.iloc[:,11:12])
X_train.iloc[:,11:12] = imputer.transform(X_train.iloc[:,11:12])
imputer = imputer.fit(X_train.iloc[:,45:46])
X_train.iloc[:,45:46] = imputer.transform(X_train.iloc[:,45:46])

#encoding categorical data
X_train = pd.get_dummies(X_train, columns=['building_id','area_assesed','land_surface_condition',
                                           'foundation_type','roof_type','ground_floor_type',
                                           'other_floor_type','position','plan_configuration',
                                           'legal_ownership_status','condition_post_eq'])
X_test = pd.get_dummies(X_test, columns=['building_id','area_assesed','land_surface_condition',
                                           'foundation_type','roof_type','ground_floor_type',
                                           'other_floor_type','position','plan_configuration',
                                           'legal_ownership_status','condition_post_eq'])

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)