import pandas as pd
df_path = r'C:\Users\28630\OneDrive\Desktop\Final Dataset.xlsx'
df = pd.read_excel(df_path)
data = df[['game_id', 'season', 'team1_score', 'team2_score', 'num_ot', 'team1_seed', 'team2_seed', 
           'T1_BPI', 'T2_BPI', 'T1_Strength of Schedule', 'T2_Strength of Schedule', 'team1_trueshootingpercentage', 'team2_trueshootingpercentage', 
           'team1_blockpct', 'team2_blockpct', 'team1_oppblockpct', 'team2_oppblockpct', 'team1_opptrueshootingpercentage', 'team2_opptrueshootingpercentage', 
           'team1_arate', 'team2_arate','team1_opparate', 'team2_opparate', 'team1_stlrate', 'team2_stlrate', 'team1_oppstlrate', 'team2_oppstlrate', 
           'combined_fatigue_team1', 'combined_fatigue_team2', 'team1_adjtempo', 'team2_adjtempo', 'team1_adjoe', 'team2_adjoe', 'team1_adjde', 'team2_adjde', 
           'W-L%_Coach_1', 'W-L%_Coach_2', 'Team1_Top_Player_Influence', 'Team2_Top_Player_Influence']]
data.shape


data['score_diff'] = data['team1_score'] - data['team2_score']
data['winner'] = data['score_diff'].apply(lambda x: 1 if x > 0 else 0)
data['num_over_time'] = data['num_ot']
data['seed_diff'] = data['team1_seed'] - data['team2_seed']
data['BPI_diff'] = data['T1_BPI'] - data['T2_BPI']
data['Strength_of_Schedule_diff'] = data['T1_Strength of Schedule'] - data['T2_Strength of Schedule']
data['trueshootingpercentage_diff'] = data['team1_trueshootingpercentage'] - data['team2_trueshootingpercentage']
data['blockpct_diff'] = data['team1_blockpct'] - data['team2_blockpct']
data['oppblockpct_diff'] = data['team1_oppblockpct'] - data['team2_oppblockpct']
data['opptrueshootingpercentage_diff'] = data['team1_opptrueshootingpercentage'] - data['team2_opptrueshootingpercentage']
data['arate_diff'] = data['team1_arate'] - data['team2_arate']
data['opparate_diff'] = data['team1_opparate'] - data['team2_opparate']
data['stlrate_diff'] = data['team1_stlrate'] - data['team2_stlrate']
data['oppstlrate_diff'] = data['team1_oppstlrate'] - data['team2_oppstlrate']
data['combined_fatigue_diff'] = data['combined_fatigue_team1'] - data['combined_fatigue_team2']
data['adjtempo_diff'] = data['team1_adjtempo'] - data['team2_adjtempo']
data['adjoe_diff'] = data['team1_adjoe'] - data['team2_adjoe']
data['adjde_diff'] = data['team1_adjde'] - data['team2_adjde']
data['W-L%_Coach_diff'] = data['W-L%_Coach_1'] - data['W-L%_Coach_2']
data['Top_Player_Influence_diff'] = data['Team1_Top_Player_Influence'] - data['Team2_Top_Player_Influence']

data.shape
data_clean = data.drop(data.columns[2:40], axis = 1)
data_clean.columns
data_clean.isna().any().any()
data1 = data_clean.sample(frac = 0.5, random_state = 88)

data2 = data_clean[~data_clean.index.isin(data1.index)].reset_index(drop = True)
data2['winner'] = 0
data2['seed_diff'] = data2['seed_diff'] * (-1)
data2['BPI_diff'] = data2['BPI_diff'] * (-1)
data2['Strength_of_Schedule_diff'] = data2['Strength_of_Schedule_diff'] * (-1)
data2['trueshootingpercentage_diff'] = data2['trueshootingpercentage_diff'] * (-1)
data2['blockpct_diff'] = data2['blockpct_diff'] * (-1)
data2['oppblockpct_diff'] = data2['oppblockpct_diff'] * (-1)
data2['opptrueshootingpercentage_diff'] = data2['opptrueshootingpercentage_diff'] * (-1)
data2['arate_diff'] = data2['arate_diff'] * (-1)
data2['opparate_diff'] = data2['opparate_diff'] * (-1)
data2['stlrate_diff'] = data2['stlrate_diff'] * (-1)
data2['oppstlrate_diff'] = data2['oppstlrate_diff'] * (-1)
data2['combined_fatigue_diff'] = data2['combined_fatigue_diff'] * (-1)
data2['adjtempo_diff'] = data2['adjtempo_diff'] * (-1)
data2['adjoe_diff'] = data2['adjoe_diff'] * (-1)
data2['adjde_diff'] = data2['adjde_diff'] * (-1)
data2['W-L%_Coach_diff'] = data2['W-L%_Coach_diff'] * (-1)
data2['Top_Player_Influence_diff'] = data2['Top_Player_Influence_diff'] * (-1)
data2['game_id'] = (data2['game_id'].str.split('-', expand = True)[0] + '-' + data2['game_id'].str.split('-', expand = True)[2] + '-' + data2['game_id'].str.split('-', expand = True)[1])

df_clean = pd.concat([data1, data2]).reset_index(drop = True)
df_clean.columns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_x = df_clean.iloc[:, 3:]
df_other = df_clean.iloc[:, 0:3]
x_scaled = scaler.fit_transform(df_x)
df_x_scaled = pd.DataFrame(x_scaled, columns = df_x.columns, index = df_x.index)
data_clean = pd.concat([df_other, df_x_scaled], axis = 1)

data_clean.shape
# Feature Selection (By using Random Forest)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

x = data_clean.drop(data_clean.iloc[:, 0:3], axis = 1)
y = data_clean['winner']
model = RandomForestClassifier(n_estimators = 100, random_state = 88) 
model.fit(x, y)

feature_importance = model.feature_importances_
features = np.array(x.columns)
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by = 'Importance', ascending = False)
importance_df

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()
plt.show()

selected_feature10 = importance_df.iloc[0:10]['Feature'].tolist()
data_clean = pd.concat([data_clean.iloc[:, 0:3], data_clean[selected_feature10]], axis = 1)


data_clean.columns





### Gaussain Naive Basys
from sklearn.naive_bayes import GaussianNB

# Rolling Window
# Define the function of rolling window
from sklearn.metrics import accuracy_score, log_loss

def rolling_window(window_duration, test_duration, df, model):
    years = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024] 
    winner_prediction_pro = []
    y_train_true = []
    winner_true = []
    y_test_pre_l = []
    y_train_pre_l = []
    loss = []
    test_acc = []
    train_acc = []

    
    for i in range(len(years) - window_duration - test_duration + 1):
        train_years = years[i : i + window_duration]  # Select training years
        test_years = years[i + window_duration : i + window_duration + test_duration]  # Select test years

        x_train = df[df['season'].isin(train_years)].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_train = df[df['season'].isin(train_years)]['winner']

        x_test = df[df['season'].isin(test_years)].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_test = df[df['season'].isin(test_years)]['winner']

        model.fit(x_train, y_train)

        # Predict probabilities
        y_test_pre = model.predict(x_test)
        y_pred_pro = model.predict_proba(x_test)[:, 1]
        y_train_pre = model.predict(x_train)

        # Append properly
        y_train_true.extend(list(y_train))  
        winner_true.extend(list(y_test))  
        y_train_pre_l.extend(list(y_train_pre))
        y_test_pre_l.extend(list(y_test_pre))  
        winner_prediction_pro.extend(list(y_pred_pro))

        
        loss.append(log_loss(winner_true, winner_prediction_pro))
        test_acc.append(accuracy_score(winner_true, y_test_pre_l))
        train_acc.append(accuracy_score(y_train_true, y_train_pre_l))

    return loss, test_acc, train_acc

# Develop a function to get all the loss
def get_all_loss(df, model):
    year = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for i in range(1, len(year)):
        test_duration = i
        for j in range(i, len(year) + 1 - i):
            window_duration = j
            if j >= i:
                loss, test_accuracy, train_accuracy = rolling_window(window_duration, test_duration, df, model)
                loss_list.extend(loss)
                train_acc_list.extend(train_accuracy)
                test_acc_list.extend(test_accuracy)
    return (loss_list, train_acc_list, test_acc_list)

loss_list_gbn, train_accuracy_list_gbn, test_accuracy_list_gbn = get_all_loss(data_clean, GaussianNB())
acc_diff_list_gbn = np.array(train_accuracy_list_gbn) - np.array(test_accuracy_list_gbn)
acc_diff_list_gbn = acc_diff_list_gbn.tolist()

loss_index_name = ['loss111', 'loss112', 'loss113', 'loss114', 'loss115', 'loss116', 'loss117', 'loss118', 
                   'loss211', 'loss212', 'loss213', 'loss214', 'loss215', 'loss216', 'loss217',
                   'loss311', 'loss312', 'loss313', 'loss314', 'loss315', 'loss316', 
                   'loss411', 'loss412', 'loss413', 'loss414', 'loss415',
                   'loss511', 'loss512', 'loss513', 'loss514', 
                   'loss611', 'loss612', 'loss613',
                   'loss711', 'loss712', 
                   'loss811', 
                   'loss221', 'loss222', 'loss223', 'loss224', 'loss225', 'loss226', 
                   'loss321', 'loss322', 'loss323', 'loss324', 'loss325', 
                   'loss421', 'loss422', 'loss423', 'loss424', 
                   'loss521', 'loss522', 'loss523',
                   'loss621', 'loss622', 
                   'loss721', 
                   'loss331', 'loss332', 'loss333', 'loss334', 
                   'loss431', 'loss432', 'loss433', 
                   'loss531', 'loss532', 
                   'loss631', 
                   'loss441', 'loss442', 
                   'loss541']
accuracy_index_name = ['acc111', 'acc112', 'acc113', 'acc114', 'acc115', 'acc116', 'acc117', 'acc118', 
                   'acc211', 'acc212', 'acc213', 'acc214', 'acc215', 'acc216', 'acc217',
                   'acc311', 'acc312', 'acc313', 'acc314', 'acc315', 'acc316', 
                   'acc411', 'acc412', 'acc413', 'acc414', 'acc415',
                   'acc511', 'acc512', 'acc513', 'acc514', 
                   'acc611', 'acc612', 'acc613',
                   'acc711', 'acc712', 
                   'acc811', 
                   'acc221', 'acc222', 'acc223', 'acc224', 'acc225', 'acc226', 
                   'acc321', 'acc322', 'acc323', 'acc324', 'acc325', 
                   'acc421', 'acc422', 'acc423', 'acc424', 
                   'acc521', 'acc522', 'acc523',
                   'acc621', 'acc622', 
                   'acc721', 
                   'acc331', 'acc332', 'acc333', 'acc334', 
                   'acc431', 'acc432', 'acc433', 
                   'acc531', 'acc532', 
                   'acc631', 
                   'acc441', 'acc442', 
                   'acc541']
df_loss_gbn = pd.DataFrame(loss_list_gbn, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_gbn = pd.DataFrame(test_accuracy_list_gbn, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_gbn = pd.DataFrame(acc_diff_list_gbn, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

index_eva = ['111', '112', '113', '114', '115', '116', '117', '118', 
                   '211', '212', '213', '214', '215', '216', '217',
                   '311', '312', '313', '314', '315', '316', 
                   '411', '412', '413', '414', '415',
                   '511', '512', '513', '514', 
                   '611', '612', '613',
                   '711', '712', 
                   '811', 
                   '221', '222', '223', '224', '225', '226', 
                   '321', '322', '323', '324', '325', 
                   '421', '422', '423', '424', 
                   '521', '522', '523',
                   '621', '622', 
                   '721', 
                   '331', '332', '333', '334', 
                   '431', '432', '433', 
                   '531', '532', 
                   '631', 
                   '441', '442', 
                   '541']
df_gnb_evaluation = pd.concat([df_loss_gbn.reset_index(drop = True), df_accuracy_gbn.reset_index(drop = True), df_acc_diff_gbn.reset_index(drop = True)], axis = 1)
df_gnb_evaluation.index = index_eva

# We assume that only 2023 and 2024 are more relevant / similar to 2025.
selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_gnb_evaluation = df_gnb_evaluation.loc[selection_year]


df_gnb_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_gnb_evaluation.sort_values(by = 'Accuracy Value', ascending = False)


df_gnb_evaluation.to_csv(r'C:\Users\28630\OneDrive\Desktop\gnb_result.csv')

# Create Plot to select the best rolling window
df_loss_gbn_sorted = df_loss_gbn.sort_values(by = 'Log Loss Value')
df_accuracy_gbn_sorted = df_accuracy_gbn.sort_values(by = 'Accuracy Value', ascending = False)
df_acc_diff_gbn_sorted = df_acc_diff_gbn.sort_values(by = 'Train-Test Accuracy Difference Value')

plt.figure(figsize = (12, 8))
bars = plt.bar(df_loss_gbn_sorted.index, df_loss_gbn_sorted['Log Loss Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_accuracy_gbn_sorted.index, df_accuracy_gbn_sorted['Accuracy Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Accuracy Value')
plt.title('Accuracy Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_acc_diff_gbn_sorted.index, df_acc_diff_gbn_sorted['Train-Test Accuracy Difference Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Train-Test Accuracy Difference Value')
plt.title('Overfitting Check')
plt.show()

# Conclusion: The Gaussain Naive Basys Model does not perform well.









# Random Forest
# Model Building
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 88)
loss_list_rf, train_accuracy_list_rf, test_accuracy_list_rf = get_all_loss(data_clean, model)
acc_diff_list_rf = np.array(train_accuracy_list_rf) - np.array(test_accuracy_list_rf)
acc_diff_list_rf = acc_diff_list_rf.tolist()

df_loss_rf = pd.DataFrame(loss_list_rf, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_rf = pd.DataFrame(test_accuracy_list_rf, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_rf = pd.DataFrame(acc_diff_list_rf, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_rf_evaluation = pd.concat([df_loss_rf.reset_index(drop = True), df_accuracy_rf.reset_index(drop = True), df_acc_diff_rf.reset_index(drop = True)], axis = 1)
df_rf_evaluation.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_rf_evaluation = df_rf_evaluation.loc[selection_year]

df_rf_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_rf_evaluation.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The Random Forest Model with these hyperparameters has the severe overfitting problem, thus try to reduce tree number and max depth.


# Random Forest Adjust Hyperparameters
# n_estimators = 10, max_depth = 5
model = RandomForestClassifier(n_estimators = 10, max_depth = 5, random_state = 88)
loss_list_rf, train_accuracy_list_rf, test_accuracy_list_rf = get_all_loss(data_clean, model)
acc_diff_list_rf = np.array(train_accuracy_list_rf) - np.array(test_accuracy_list_rf)
acc_diff_list_rf = acc_diff_list_rf.tolist()


df_loss_rf = pd.DataFrame(loss_list_rf, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_rf = pd.DataFrame(test_accuracy_list_rf, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_rf = pd.DataFrame(acc_diff_list_rf, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_rf_evaluation1 = pd.concat([df_loss_rf.reset_index(drop = True), df_accuracy_rf.reset_index(drop = True), df_acc_diff_rf.reset_index(drop = True)], axis = 1)
df_rf_evaluation1.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_rf_evaluation1 = df_rf_evaluation1.loc[selection_year]

df_rf_evaluation1.sort_values(by = 'Log Loss Value', ascending = True)
df_rf_evaluation1.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The Random Forest Model with these hyperparameters reduces the effect ofoverfitting problem, but the problem still exists. Try to reduce max depth.


# Random Forest Adjust Hyperparameters
# n_estimators = 10, max_depth = 3
model = RandomForestClassifier(n_estimators = 10, max_depth = 3, random_state = 88)
loss_list_rf, train_accuracy_list_rf, test_accuracy_list_rf = get_all_loss(data_clean, model)
acc_diff_list_rf = np.array(train_accuracy_list_rf) - np.array(test_accuracy_list_rf)
acc_diff_list_rf = acc_diff_list_rf.tolist()

df_loss_rf = pd.DataFrame(loss_list_rf, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_rf = pd.DataFrame(test_accuracy_list_rf, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_rf = pd.DataFrame(acc_diff_list_rf, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_rf_evaluation2 = pd.concat([df_loss_rf.reset_index(drop = True), df_accuracy_rf.reset_index(drop = True), df_acc_diff_rf.reset_index(drop = True)], axis = 1)
df_rf_evaluation2.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_rf_evaluation2 = df_rf_evaluation2.loc[selection_year]

df_rf_evaluation2.sort_values(by = 'Log Loss Value', ascending = True)
df_rf_evaluation2.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The Random Forest Model is better. Try to reduce tree number to see the outcome.


# Random Forest Adjust Hyperparameters
# n_estimators = 5, max_depth = 3
model = RandomForestClassifier(n_estimators = 5, max_depth = 3, random_state = 88)
loss_list_rf, train_accuracy_list_rf, test_accuracy_list_rf = get_all_loss(data_clean, model)
acc_diff_list_rf = np.array(train_accuracy_list_rf) - np.array(test_accuracy_list_rf)
acc_diff_list_rf = acc_diff_list_rf.tolist()

df_loss_rf = pd.DataFrame(loss_list_rf, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_rf = pd.DataFrame(test_accuracy_list_rf, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_rf = pd.DataFrame(acc_diff_list_rf, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_rf_evaluation3 = pd.concat([df_loss_rf.reset_index(drop = True), df_accuracy_rf.reset_index(drop = True), df_acc_diff_rf.reset_index(drop = True)], axis = 1)
df_rf_evaluation3.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_rf_evaluation3 = df_rf_evaluation3.loc[selection_year]

df_rf_evaluation3.sort_values(by = 'Log Loss Value', ascending = True)
df_rf_evaluation3.sort_values(by = 'Accuracy Value', ascending = False)

df_rf_evaluation3.to_csv(r'C:\Users\28630\OneDrive\Desktop\rf_result.csv')

# Create Plot to select the best rolling window
df_loss_rf_sorted = df_loss_rf.sort_values(by = 'Log Loss Value')
df_accuracy_rf_sorted = df_accuracy_rf.sort_values(by = 'Accuracy Value', ascending = False)
df_acc_diff_rf_sorted = df_acc_diff_rf.sort_values(by = 'Train-Test Accuracy Difference Value')

plt.figure(figsize = (12, 6))
bars = plt.bar(df_loss_rf_sorted.index, df_loss_rf_sorted['Log Loss Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_accuracy_rf_sorted.index, df_accuracy_rf_sorted['Accuracy Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Accuracy Value')
plt.title('Accuracy Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_acc_diff_rf_sorted.index, df_acc_diff_rf_sorted['Train-Test Accuracy Difference Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Train-Test Accuracy Difference Value')
plt.title('Overfitting Check')
plt.show()

# Conclusion: This model is better than the previous one. Try to explore more.


# Random Forest Adjust Hyperparameters
# n_estimators = 10, max_depth = 3
model = RandomForestClassifier(n_estimators = 10, max_depth = 3, random_state = 88)
loss_list_rf, train_accuracy_list_rf, test_accuracy_list_rf = get_all_loss(data_clean, model)
acc_diff_list_rf = np.array(train_accuracy_list_rf) - np.array(test_accuracy_list_rf)
acc_diff_list_rf = acc_diff_list_rf.tolist()


df_loss_rf = pd.DataFrame(loss_list_rf, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_rf = pd.DataFrame(test_accuracy_list_rf, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_rf = pd.DataFrame(acc_diff_list_rf, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_rf_evaluation4 = pd.concat([df_loss_rf.reset_index(drop = True), df_accuracy_rf.reset_index(drop = True), df_acc_diff_rf.reset_index(drop = True)], axis = 1)
df_rf_evaluation4.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_rf_evaluation4 = df_rf_evaluation4.loc[selection_year]

df_rf_evaluation4.sort_values(by = 'Log Loss Value', ascending = True)
df_rf_evaluation4.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: This model is slightly worse than the previous one.

# Conclusion: For Random Forest Model, choose n_estimators = 5, max_depth = 3. 
#             The best rolling window would be window_duration = 7, test_duration = 2.
#             This means the model use 2015 to 2022 data as the train data to test the 2023 and 2024 data.
  





# SVM
# Model Building
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale', random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation = df_svm_evaluation.loc[selection_year]

df_svm_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The modle has overfitting issue. Try to adjust hyperparameters to address this issue.


# Adjust Hyperparameters
# SVM: kernel = 'linear', C = 1.0
model = SVC(kernel = 'linear', C = 1.0, random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation1 = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation1.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation1 = df_svm_evaluation1.loc[selection_year]

df_svm_evaluation1.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation1.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: This model has addressed the overfitting problem and has a pretty good evaluation result.
#             Try to explore more.


# Adjust Hyperparameters
# SVM: kernel = 'linear', C = 0.1
model = SVC(kernel = 'linear', C = 0.1, random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation2 = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation2.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation2 = df_svm_evaluation2.loc[selection_year]

df_svm_evaluation2.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation2.sort_values(by = 'Accuracy Value', ascending = False)

df_svm_evaluation2.to_csv(r'C:\Users\28630\OneDrive\Desktop\svm_result.csv')

# Create Plot to select the best rolling window
df_loss_svm_sorted = df_loss_svm.sort_values(by = 'Log Loss Value')
df_accuracy_svm_sorted = df_accuracy_svm.sort_values(by = 'Accuracy Value', ascending = False)
df_acc_diff_svm_sorted = df_acc_diff_svm.sort_values(by = 'Train-Test Accuracy Difference Value')

plt.figure(figsize = (12, 6))
bars = plt.bar(df_loss_svm_sorted.index, df_loss_svm_sorted['Log Loss Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_accuracy_svm_sorted.index, df_accuracy_svm_sorted['Accuracy Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Accuracy Value')
plt.title('Accuracy Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_acc_diff_svm_sorted.index, df_acc_diff_svm_sorted['Train-Test Accuracy Difference Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Train-Test Accuracy Difference Value')
plt.title('Overfitting Check')
plt.show()

# Conclusion: This model is slightly better than the previous one. Explore more.


# Adjust Hyperparameters
# kernel = 'linear', C = 0.01
model = SVC(kernel = 'linear', C = 0.01, random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation3 = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation3.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation3 = df_svm_evaluation3.loc[selection_year]

df_svm_evaluation3.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation3.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model is worse than the previous one. Explore more.

# Adjust Hyperparameters
# kernel = 'rbf', C = 0.1, gamma = 'scale'
model = SVC(kernel = 'rbf', C = 0.1, gamma = 'scale', random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation4 = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation4.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation4 = df_svm_evaluation4.loc[selection_year]

df_svm_evaluation4.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation4.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model is worse than the df_svm_evaluation2. Explore more.


# Adjust Hyperparameters
# kernel = 'rbf', C = 0.01, gamma = 'scale'
model = SVC(kernel = 'rbf', C = 0.01, gamma = 'scale', random_state = 88, probability = True)
loss_list_svm, train_accuracy_list_svm, test_accuracy_list_svm = get_all_loss(data_clean, model)
acc_diff_list_svm = np.array(train_accuracy_list_svm) - np.array(test_accuracy_list_svm)
acc_diff_list_svm = acc_diff_list_svm.tolist()

df_loss_svm = pd.DataFrame(loss_list_svm, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_svm = pd.DataFrame(test_accuracy_list_svm, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_svm = pd.DataFrame(acc_diff_list_svm, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_svm_evaluation5 = pd.concat([df_loss_svm.reset_index(drop = True), df_accuracy_svm.reset_index(drop = True), df_acc_diff_svm.reset_index(drop = True)], axis = 1)
df_svm_evaluation5.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_svm_evaluation5 = df_svm_evaluation5.loc[selection_year]

df_svm_evaluation5.sort_values(by = 'Log Loss Value', ascending = True)
df_svm_evaluation5.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: This model is suck.

# Conclusion: For Support Vector Machine Model, choose kernal = 'linear', C = 0.1. 
#             The best rolling window would be window_duration = 8, test_duration = 1.
#             This means the model use 2015 to 2023 data as the train data to test the 2024 data.


# XG Boosting
# Model Building
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 5)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation = df_xg_evaluation.loc[selection_year]

df_xg_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model has severe overfitting problem. Try to adjust hyperparameters.


# Adjust Hyperparameters
# n_estimators = 10, learning_rate = 0.1, max_depth = 5
model = XGBClassifier(n_estimators = 10, learning_rate = 0.1, max_depth = 5)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation1 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation1.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation1 = df_xg_evaluation1.loc[selection_year]

df_xg_evaluation1.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation1.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model still has the overfitting issue. Reduce max depth


# Adjust Hyperparameters
# n_estimators = 10, learning_rate = 0.1, max_depth = 3
model = XGBClassifier(n_estimators = 10, learning_rate = 0.1, max_depth = 3)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation2 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation2.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation2 = df_xg_evaluation2.loc[selection_year]

df_xg_evaluation2.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation2.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model still has the issue. But it is not that severe. Try to reduce tree number.


# Adjust Hyperparameters
# n_estimators = 5, learning_rate = 0.1, max_depth = 3
model = XGBClassifier(n_estimators = 5, learning_rate = 0.1, max_depth = 3)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation3 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation3.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation3 = df_xg_evaluation3.loc[selection_year]

df_xg_evaluation3.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation3.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model still has overfitting issue. 

# Adjust Hyperparameters
# n_estimators = 5, learning_rate = 0.2, max_depth = 3
model = XGBClassifier(n_estimators = 5, learning_rate = 0.2, max_depth = 3)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation4 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation4.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation4 = df_xg_evaluation4.loc[selection_year]

df_xg_evaluation4.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation4.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model is better. Try to increase learning rate


# Adjust Hyperparameters
# n_estimators = 5, learning_rate = 0.3, max_depth = 3
model = XGBClassifier(n_estimators = 5, learning_rate = 0.3, max_depth = 3)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation5 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation5.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation5 = df_xg_evaluation5.loc[selection_year]

df_xg_evaluation5.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation5.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: Slightly better. Explore more


# Adjust Hyperparameters
# n_estimators = 10, learning_rate = 0.3, max_depth = 3
model = XGBClassifier(n_estimators = 10, learning_rate = 0.3, max_depth = 3)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation6 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation6.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation6 = df_xg_evaluation6.loc[selection_year]

df_xg_evaluation6.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation6.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model is slightly worth. Try to explore more.


# Adjust Hyperparameters
# n_estimators = 5, learning_rate = 0.3, max_depth = 2
model = XGBClassifier(n_estimators = 5, learning_rate = 0.3, max_depth = 2)
loss_list_xg, train_accuracy_list_xg, test_accuracy_list_xg = get_all_loss(data_clean, model)
acc_diff_list_xg = np.array(train_accuracy_list_xg) - np.array(test_accuracy_list_xg)
acc_diff_list_xg = acc_diff_list_xg.tolist()

df_loss_xg = pd.DataFrame(loss_list_xg, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_xg = pd.DataFrame(test_accuracy_list_xg, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_xg = pd.DataFrame(acc_diff_list_xg, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_xg_evaluation7 = pd.concat([df_loss_xg.reset_index(drop = True), df_accuracy_xg.reset_index(drop = True), df_acc_diff_xg.reset_index(drop = True)], axis = 1)
df_xg_evaluation7.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_xg_evaluation7 = df_xg_evaluation7.loc[selection_year]

df_xg_evaluation7.sort_values(by = 'Log Loss Value', ascending = True)
df_xg_evaluation7.sort_values(by = 'Accuracy Value', ascending = False)

df_xg_evaluation7.to_csv(r'C:\Users\28630\OneDrive\Desktop\xg_result.csv')

# Create Plot to select the best rolling window
df_loss_xg_sorted = df_loss_xg.sort_values(by = 'Log Loss Value')
df_accuracy_xg_sorted = df_accuracy_xg.sort_values(by = 'Accuracy Value', ascending = False)
df_acc_diff_xg_sorted = df_acc_diff_xg.sort_values(by = 'Train-Test Accuracy Difference Value')

plt.figure(figsize = (12, 6))
bars = plt.bar(df_loss_xg_sorted.index, df_loss_xg_sorted['Log Loss Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_accuracy_xg_sorted.index, df_accuracy_xg_sorted['Accuracy Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Accuracy Value')
plt.title('Accuracy Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_acc_diff_xg_sorted.index, df_acc_diff_xg_sorted['Train-Test Accuracy Difference Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Train-Test Accuracy Difference Value')
plt.title('Overfitting Check')
plt.show()

# Conclusion: The model is slightly better. 



# Conclusion: For the best XG Boosting Model, the hyperparameters should be 
#             n_estimators = 5, learning_rate = 0.3, max_depth = 2. The best rolling window is
#             window_duration = 8, test_duration = 1. This means the model use 2015 to 2023 data
#             to test the 2024 data








# Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty = 'l2', C = 1)
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation = df_lr_evaluation.loc[selection_year]

df_lr_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation.sort_values(by = 'Accuracy Value', ascending = False)

df_lr_evaluation.to_csv(r'C:\Users\28630\OneDrive\Desktop\lr_result.csv')

# Conclusion: The model performs well. Try to adjust hyperparameters to see if it can have
#             a better performance.


# Adjust Hyperparameters
# penalty = 'l1', C = 1, solver = 'saga'
model = LogisticRegression(penalty = 'l1', C = 1, solver = 'saga')
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation1 = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation1.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation1 = df_lr_evaluation1.loc[selection_year]

df_lr_evaluation1.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation1.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: Good model.


# Adjust Hyperparameters
# penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, C = 1.0
model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, C = 1)
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation2 = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation2.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation2 = df_lr_evaluation2.loc[selection_year]

df_lr_evaluation2.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation2.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: The model is slightly worse than df_lr_evaluation. Explore more.


# Adjust Hyperparameters
# penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.3, C = 1
model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.3, C = 1)
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation3 = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation3.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation3 = df_lr_evaluation3.loc[selection_year]

df_lr_evaluation3.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation3.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: Good Model


# Adjust Hyperparameters
# penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.7, C = 1
model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.7, C = 1)
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation4 = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation4.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation4 = df_lr_evaluation4.loc[selection_year]

df_lr_evaluation4.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation4.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: Slightly worse


# Adjust Hyperparameters
# penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.2, C = 1
model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.2, C = 1)
loss_list_lr, train_accuracy_list_lr, test_accuracy_list_lr = get_all_loss(data_clean, model)
acc_diff_list_lr = np.array(train_accuracy_list_lr) - np.array(test_accuracy_list_lr)
acc_diff_list_lr = acc_diff_list_lr.tolist()

df_loss_lr = pd.DataFrame(loss_list_lr, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_lr = pd.DataFrame(test_accuracy_list_lr, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_lr = pd.DataFrame(acc_diff_list_lr, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

df_lr_evaluation5 = pd.concat([df_loss_lr.reset_index(drop = True), df_accuracy_lr.reset_index(drop = True), df_acc_diff_lr.reset_index(drop = True)], axis = 1)
df_lr_evaluation5.index = index_eva

selection_year = ['118', '217', '316', '415', '514', '613', '712', '811', '226', '325', '424', '523', '622', '721'] 
df_lr_evaluation5 = df_lr_evaluation5.loc[selection_year]

df_lr_evaluation5.sort_values(by = 'Log Loss Value', ascending = True)
df_lr_evaluation5.sort_values(by = 'Accuracy Value', ascending = False)

# Conclusion: No obvious difference when comparing with the previous one.


# Conclusion: For the best Logistics Regression, the hyperparameters should be 
#             penalty = 'l2', C = 1.
#             And the best rolling window is window_duration = 8, test_duration = 1.
#             This means the model use 2015 to 2023 data to test the 2024 data.




### Bayesian Logistic Regression Model
import pymc as pm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_x = df_clean.iloc[:, 3:]
df_other = df_clean.iloc[:, 0:3]
x_scaled = scaler.fit_transform(df_x)
df_x_scaled = pd.DataFrame(x_scaled, columns = df_x.columns, index = df_x.index)
data_clean_baye = pd.concat([df_other, df_x_scaled], axis=1)


def rolling_window_baye(df):
    train_year = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024] 
    winner_prediction_pro = []
    y_train_true = []
    winner_true = []
    y_test_pre_l = []
    y_train_pre_l = []
    loss = []
    test_acc = []
    train_acc = []

    # test_year= 2024
    for i in range(0, 8):
        x_test = df[df['season'] == 2024].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_test = df[df['season'] == 2024]['winner']

        x_train = df[df['season'].isin(train_year[(8-i-1):8])].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_train = df[df['season'].isin(train_year[(8-i-1):8])]['winner']

        with pm.Model() as model:
            # Priors for coefficients
            beta = pm.Normal('beta', mu = 0, sigma = 1, shape = x_train.shape[1])
            intercept = pm.Normal("intercept", mu = 0, sigma = 1)

            # Logistic function for probability
            logits = pm.math.dot(x_train, beta) + intercept
            probability = pm.Deterministic('probability', pm.math.sigmoid(logits))
    
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', p = probability, observed = y_train)

            # Sampling from the posterior
            trace = pm.sample(2000, return_inferencedata = True, target_accept = 0.9)
        
        beta_post = trace.posterior["beta"].mean(dim = ("chain", "draw")).values
        intercept_post = trace.posterior["intercept"].mean(dim = ("chain", "draw")).values
        
        logits_test = np.dot(x_test, beta_post) + intercept_post
        logits_train = np.dot(x_train, beta_post) + intercept_post

        p_pred_test = 1 / (1 + np.exp(-logits_test))  
        p_pred_train = 1 / (1 + np.exp(-logits_train))

        y_pred_test = (p_pred_test > 0.5).astype(int)
        y_pred_train = (p_pred_train > 0.5).astype(int)

        y_train_true.extend(list(y_train))
        winner_prediction_pro.extend(list(p_pred_test))
        winner_true.extend(list(y_test))
        y_test_pre_l.extend(list(y_pred_test))
        y_train_pre_l.extend(list(y_pred_train))
        
        loss.append(log_loss(winner_true, winner_prediction_pro))
        train_acc.append(accuracy_score(y_train_true, y_train_pre_l))
        test_acc.append(accuracy_score(winner_true, y_test_pre_l))

    # test_year=[2023,2024]
    for i in range(0, 7):
        x_test = df[df['season'].isin([2023,2024])].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_test = df[df['season'].isin([2023,2024])]['winner']

        x_train = df[df['season'].isin(train_year[(7-i-1):7])].drop(columns=['season', 'winner', 'game_id'], errors='ignore')
        y_train = df[df['season'].isin(train_year[(7-i-1):7])]['winner']

        with pm.Model() as model:
            # Priors for coefficients
            beta = pm.Normal('beta', mu = 0, sigma = 1, shape = x_train.shape[1])
            intercept = pm.Normal("intercept", mu = 0, sigma = 1)

            # Logistic function for probability
            logits = pm.math.dot(x_train, beta) + intercept
            probability = pm.Deterministic('probability', pm.math.sigmoid(logits))
    
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', p = probability, observed = y_train)

            # Sampling from the posterior
            trace = pm.sample(2000, return_inferencedata = True, target_accept = 0.9)
        
        beta_post = trace.posterior["beta"].mean(dim = ("chain", "draw")).values
        intercept_post = trace.posterior["intercept"].mean(dim = ("chain", "draw")).values
        
        logits_test = np.dot(x_test, beta_post) + intercept_post
        logits_train = np.dot(x_train, beta_post) + intercept_post

        p_pred_test = 1 / (1 + np.exp(-logits_test))  
        p_pred_train = 1 / (1 + np.exp(-logits_train))

        y_pred_test = (p_pred_test > 0.5).astype(int)
        y_pred_train = (p_pred_train > 0.5).astype(int)

        y_train_true.extend(list(y_train))
        winner_prediction_pro.extend(list(p_pred_test))
        winner_true.extend(list(y_test))
        y_test_pre_l.extend(list(y_pred_test))
        y_train_pre_l.extend(list(y_pred_train))
        
        loss.append(log_loss(winner_true, winner_prediction_pro))
        train_acc.append(accuracy_score(y_train_true, y_train_pre_l))
        test_acc.append(accuracy_score(winner_true, y_test_pre_l))

    return (loss, train_acc, test_acc)

loss_list_baye, train_accuracy_list_baye, test_accuracy_list_baye = rolling_window_baye(data_clean_baye)
acc_diff_list_baye = np.array(train_accuracy_list_baye) - np.array(test_accuracy_list_baye)
acc_diff_list_baye = acc_diff_list_baye.tolist()
acc_diff_list_baye

loss_index_name = ['loss118', 'loss217', 'loss316', 'loss415', 'loss514', 'loss613', 'loss712', 'loss811', 'loss216', 'loss226', 'loss325', 'loss424', 'loss523', 'loss622', 'loss721']
accuracy_index_name = ['acc118', 'acc217', 'acc316', 'acc415', 'acc514', 'acc613', 'acc712', 'acc811', 'acc216', 'acc226', 'acc325', 'acc24', 'acc523', 'acc622', 'accu721']
df_loss_baye = pd.DataFrame(loss_list_baye, index = loss_index_name, columns = ['Log Loss Value'])
df_accuracy_baye = pd.DataFrame(test_accuracy_list_baye, index = accuracy_index_name, columns = ['Accuracy Value'])
df_acc_diff_baye = pd.DataFrame(acc_diff_list_baye, index = accuracy_index_name, columns = ['Train-Test Accuracy Difference Value'])

index_eva = ['118', '217', '316', '415', '514', '613', '712', '811', '216', '226', '325', '424', '523', '622', '721']
df_baye_evaluation = pd.concat([df_loss_baye.reset_index(drop = True), df_accuracy_baye.reset_index(drop = True), df_acc_diff_baye.reset_index(drop = True)], axis = 1)
df_baye_evaluation.index = index_eva
df_baye_evaluation = df_baye_evaluation.drop(index = '216')

df_baye_evaluation.sort_values(by = 'Log Loss Value', ascending = True)
df_baye_evaluation.sort_values(by = 'Accuracy Value', ascending = False)



df_baye_evaluation.to_csv(r'C:\Users\28630\OneDrive\Desktop\Baye_result.csv')

# Create Plot to select the best rolling window
df_loss_baye_sorted = df_loss_baye.sort_values(by = 'Log Loss Value')
df_accuracy_baye_sorted = df_accuracy_baye.sort_values(by = 'Accuracy Value', ascending = False)
df_acc_diff_baye_sorted = df_acc_diff_baye.sort_values(by = 'Train-Test Accuracy Difference Value')

plt.figure(figsize = (12, 8))
bars = plt.bar(df_loss_baye_sorted.index, df_loss_baye_sorted['Log Loss Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Log Loss Value')
plt.title('Log Loss Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_accuracy_baye_sorted.index, df_accuracy_baye_sorted['Accuracy Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Accuracy Value')
plt.title('Accuracy Comparison')
plt.show()

plt.figure(figsize = (12, 6))
bars = plt.bar(df_acc_diff_baye_sorted.index, df_acc_diff_baye_sorted['Train-Test Accuracy Difference Value'], color = 'skyblue')
for bar in bars:
    yval = bar.get_height()  
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha = 'center', va = 'bottom', fontsize = 10)
plt.xticks(rotation = 45) 
plt.xlabel('Rolling Window Types')
plt.ylabel('Train-Test Accuracy Difference Value')
plt.title('Overfitting Check')
plt.show()

# Conclusion: The model has the moderate overfitting problem.

# Conclusion: The best rolling window is window_duration = 8, test_duration = 1. 
#             This means that the model use 2015 and 2023 data to test the 2024 data.