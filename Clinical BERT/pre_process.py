import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

csv_path = r'C:\Users\krish\Desktop\OMSA\Spring 2022\CSE 6250 - BD4H\Project\Data'

# Load ADMISSIONS Table
df_admission = pd.read_csv(os.path.join(csv_path, 'ADMISSIONS.csv'))

# Converting Strings to Dates.
df_admission['ADMITTIME'] = pd.to_datetime(df_admission['ADMITTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_admission['DISCHTIME'] = pd.to_datetime(df_admission['DISCHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_admission['DEATHTIME'] = pd.to_datetime(df_admission['DEATHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

df_admission = df_admission.sort_values(['SUBJECT_ID', 'ADMITTIME'])
df_admission = df_admission.reset_index(drop=True)
df_admission['NEXT_ADMITTIME'] = df_admission.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_admission['NEXT_ADMISSION_TYPE'] = df_admission.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

rows = df_admission['NEXT_ADMISSION_TYPE'] == 'ELECTIVE'
df_admission.loc[rows,'NEXT_ADMITTIME'] = np.NaN
df_admission.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

df_admission = df_admission.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admission[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_admission.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

# Days until next admission
df_admission['DAYS_NEXT_ADMIT'] = (df_admission.NEXT_ADMITTIME - df_admission.DISCHTIME).dt.total_seconds()/(24*60*60)

# Removing Newborn Admissions and admissions with deathtime - Since we are concerned with Emergency and Urgent admissions, we remove newborns and dead patients
df_admission = df_admission.loc[df_admission['ADMISSION_TYPE'] != 'NEWBORN']
df_admission = df_admission.loc[df_admission['DEATHTIME'].isnull()]

# To classify whether a patient will be readmitted in the next 30 days, wecreate a output variable to classify as follows: 1 = readmitted, 0 = not readmitted
df_admission['OUTPUT_LABEL'] = (df_admission['DAYS_NEXT_ADMIT'] < 30).astype('int')
df_admission['DURATION'] = (df_admission['DISCHTIME'] - df_admission['ADMITTIME']).dt.total_seconds()/(24*60*60)

# Load NOTEEVENTS Table
df_notev = pd.read_csv(os.path.join(csv_path, 'NOTEEVENTS.csv'))
df_notev = df_notev.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
df_admn_notev = pd.merge(
    df_admission[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','OUTPUT_LABEL','DURATION']],
    df_notev[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY']], on = ['SUBJECT_ID','HADM_ID'], how = 'left'
)

df_admn_notev['ADMITTIME_C'] = df_admn_notev['ADMITTIME'].apply(lambda x: str(x).split(' ')[0])

df_admn_notev['ADMITTIME_C'] = pd.to_datetime(df_admn_notev['ADMITTIME_C'], format = '%Y-%m-%d', errors = 'coerce')
df_admn_notev['CHARTDATE'] = pd.to_datetime(df_admn_notev['CHARTDATE'], format = '%Y-%m-%d', errors = 'coerce')

# Discharge Summary
df_discharge = df_admn_notev[df_admn_notev['CATEGORY'] == 'Discharge summary']
df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
df_discharge = df_discharge[df_discharge['TEXT'].notnull()]

# Early bites (Less than n days on the admission notes)
def func_n_days_less(df_admn_notev, n):
    df1 = df_admn_notev[((df_admn_notev['CHARTDATE'] - df_admn_notev['ADMITTIME_C']).dt.total_seconds()/(24*60*60))<n]
    df1 = df1[df1['TEXT'].notnull()]

    df = pd.DataFrame(df1.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df['OUTPUT_LABEL'] = df['HADM_ID'].apply(lambda x: df1[df1['HADM_ID'] == x].OUTPUT_LABEL.values[0])
    return df

df_2_less = func_n_days_less(df_admn_notev, 2)
df_3_less = func_n_days_less(df_admn_notev, 3)

def regex_preprocess(x_in):
    reg = re.sub('\\[(.*?)\\]', '', x_in)  # remove de-identified brackets
    reg = re.sub('[0-9]+\.', '', reg)  # remove 1.2. since the segmenter segments based on this
    reg = re.sub('dr\.', 'doctor', reg)
    reg = re.sub('m\.d\.', 'md', reg)
    reg = re.sub('admission date:', '', reg)
    reg = re.sub('discharge date:', '', reg)
    reg = re.sub('--|__|==', '', reg)
    return reg

def preprocessing_chunks(n_less_df):
    n_less_df['TEXT'] = n_less_df['TEXT'].fillna(' ')
    n_less_df['TEXT'] = n_less_df['TEXT'].str.replace('\n', ' ')
    n_less_df['TEXT'] = n_less_df['TEXT'].str.replace('\r', ' ')
    n_less_df['TEXT'] = n_less_df['TEXT'].apply(str.strip)
    n_less_df['TEXT'] = n_less_df['TEXT'].str.lower()
    n_less_df['TEXT'] = n_less_df['TEXT'].apply(lambda x: regex_preprocess(x))

    wc = pd.DataFrame({'ID': [], 'TEXT': [], 'Label': []})
    for i in tqdm(range(len(n_less_df))):
        x = n_less_df.TEXT.iloc[i].split()
        n = int(len(x) / 318)
        for j in range(n):
            wc = wc.append({'TEXT': ' '.join(x[j * 318:(j + 1) * 318]), 'Label': n_less_df['OUTPUT_LABEL'].iloc[i],
                                'ID': n_less_df.HADM_ID.iloc[i]}, ignore_index=True)
        if len(x) % 318 > 10:
            wc = wc.append({'TEXT': ' '.join(x[-(len(x) % 318):]), 'Label': n_less_df['OUTPUT_LABEL'].iloc[i],
                                'ID': n_less_df['HADM_ID'].iloc[i]}, ignore_index=True)

    return wc

# df_discharge = preprocessing_chunks(df_discharge)
# df_2_less = preprocessing_chunks(df_2_less)
# df_3_less = preprocessing_chunks(df_3_less)

# The above process took about 5 hours on my laptop. I am going to pickle the files for later use.

# df_discharge.to_pickle("./pickle/df_discharge.pkl")
# df_2_less.to_pickle("./pickle/df_2_less.pkl")
# df_3_less.to_pickle("./pickle/df_3_less.pkl")

# Loading Pickled files
df_discharge = pd.read_pickle('./pickle/df_discharge.pkl')
df_2_less = pd.read_pickle('./pickle/df_2_less.pkl')
df_3_less = pd.read_pickle('./pickle/df_3_less.pkl')

print(df_discharge.shape) # Yields (216954, 3)
print(df_2_less.shape) # Yields (277443, 3)
print(df_3_less.shape) # Yields (385724, 3)

# TRAIN/TEST/SPLIT
readmit_ID = df_admission[df_admission['OUTPUT_LABEL'] == 1]['HADM_ID']
not_readmit_ID = df_admission[df_admission['OUTPUT_LABEL'] == 0]['HADM_ID']
not_readmit_ID_use = not_readmit_ID.sample(n=len(readmit_ID), random_state=1)
id_val_test_t = readmit_ID.sample(frac=0.2, random_state=1)
id_val_test_f = not_readmit_ID_use.sample(frac=0.2, random_state=1)
id_train_t = readmit_ID.drop(id_val_test_t.index)
id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)
id_val_t = id_val_test_t.sample(frac=0.5, random_state=1)
id_test_t = id_val_test_t.drop(id_val_t.index)
id_val_f = id_val_test_f.sample(frac=0.5, random_state=1)
id_test_f = id_val_test_f.drop(id_val_f.index)

(pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values # check for overlap between train and test, should return "array([], dtype=int64)"

id_test = pd.concat([id_test_t, id_test_f])
test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
                             columns=['id', 'label'])

id_val = pd.concat([id_val_t, id_val_f])
val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))), columns=['id', 'label'])

id_train = pd.concat([id_train_t, id_train_f])
train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
                              columns=['id', 'label'])

discharge_train = df_discharge[df_discharge['ID'].isin(train_id_label['id'])]
discharge_valid = df_discharge[df_discharge['ID'].isin(val_id_label['id'])]
discharge_test = df_discharge[df_discharge['ID'].isin(test_id_label['id'])]

# Subsampling for training - df_discharge
df = pd.concat([not_readmit_ID_use, not_readmit_ID])
df = df.drop_duplicates(keep=False)
(pd.Index(df).intersection(pd.Index(not_readmit_ID_use))).values
not_readmit_ID_more = df.sample(n=400, random_state=1)
discharge_train_snippets = pd.concat([df_discharge[df_discharge['ID'].isin(not_readmit_ID_more)], discharge_train])
discharge_train_snippets = discharge_train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)
discharge_train_snippets.Label.value_counts()

discharge_train_snippets.to_csv('./data/discharge/train.csv')
discharge_valid.to_csv('./data/discharge/validate.csv')
discharge_test.to_csv('./data/discharge/test.csv')

df_2_less = pd.read_pickle('./pickle/df_3_less.pkl')

# Subsampling for training - df_3_less
early_train = df_3_less[df_3_less['ID'].isin(train_id_label['id'])]
not_readmit_ID_more = df.sample(n=500, random_state=1)
early_train_snippets = pd.concat([df_3_less[df_3_less['ID'].isin(not_readmit_ID_more)], early_train])
early_train_snippets = early_train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)
early_train_snippets.to_csv('./data/less3/train.csv')
early_valid = df_3_less[df_3_less['ID'].isin(val_id_label['id'])]
early_valid.to_csv('./data/less3/validate.csv')
actionable_ID_3days = df_admission[df_admission['DURATION'] >= 3].HADM_ID
test_actionable_id_label = test_id_label[test_id_label['id'].isin(actionable_ID_3days)]
early_test = df_3_less[df_3_less['ID'].isin(test_actionable_id_label['id'])]
early_test.to_csv('./data/less3/test.csv')

# Subsampling for test - df_2_less
actionable_ID_2days = df_admission[df_admission['DURATION'] >= 2]['HADM_ID']
test_actionable_id_label_2days = test_id_label[test_id_label['id'].isin(actionable_ID_2days)]
early_test_2days = df_2_less[df_2_less['ID'].isin(test_actionable_id_label_2days['id'])]
early_test_2days.to_csv('./data/less2/test.csv')

