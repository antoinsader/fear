
# ## Purpose
# 
# In this file, we will clean the two datasets used in the paper Humans display interindividual differences in the latent mechanisms underlying fear generalization behaviour. We will also generate the required data structures for modeling and create some visualizations to better understand the fundamental structure of the data.
# 
# If you download everything on the OSF, you can open the markdown files with the R project - Multiplaths_Generalization.Rproj. By doing so, you won’t need to change the depository of, for example, model scripts and data files.


# ## 1. Dataset 1: Simple conditioning


# ### 1.1 Pre-process
# 


# This code chunk performs the following tasks:
# 
# It creates a list of data files for the first experiment, excluding the data for participants 15, 17, and 31.
# 
# It loads the data and processes it by:
# 
# Creating a stimulus column and replacing 999 values with NA
# Creating a stimulus_phy column
# Removing practice trials and ITI trials
# Creating a CStrials and CS_phy column
# Renaming the Size column to Per_size and selecting the desired columns
# It creates a Phy_size column.
# It changes the levels of the stimulus and stimulus_phy columns.
# 
# It creates a trials column.


import pandas as pd
import numpy as np
import pickle
from natsort import natsorted

root_folder = '../../'

# Configuration:


Nactrials_1 = 14
Nactrials_2 = 24
Ngetrials_1 = 174
Ngetrials_2 = 156
Ntrials_1 = Nactrials_1 + Ngetrials_1
Ntrials_2 = Nactrials_2 + Ngetrials_2


start= 50.8
end = 119.42
step = 7.624

# Create a sequence of stimulus sizes ranging from 50.80 to 119.42 with a step size of 7.624.
stimulus_size =  np.round(np.arange(start, end + step, step), 2)

# Create a list of stimulus levels for stimulus_level_1 including CS+
stimulus_level_1 = ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"]

# Create a vector of stimulus levels for stimulus_level_2. including cs+ and cs-
stimulus_level_2 = ["CS+","S2","S3","S4","S5","S6","S7","S8","S9","CS-"]



# Load data of experiment 1:


# Prepare data:


# Create a list of data files for the first experiment.
# The data for participants 15, 17, and 31 are broken, so they are excluded from the list.
participants = list(range(1, 15)) + list(range(16, 31)) + list(range(32, 37)) + list(range(38, 44))
data_s1_list = [f"{root_folder}Data/Experiment_1/{p}/{p}_results.txt" for p in participants]
data_s1 = (
    pd.concat(
        [pd.read_csv(file, sep="\t").assign(participant=participant) 
         for participant, file in enumerate(data_s1_list, start=1)],
        ignore_index=True
    )
)

# Create a stimulus column and replace 999 values with NaN
data_s1['stimulus'] = data_s1.apply(lambda row: 'S4' if row['C4'] == 1
                                                else 'S5' if row['C5'] == 1
                                                else 'S6' if row['C6'] == 1
                                                else 'CS+' if row['C7'] == 1
                                                else 'S8' if row['C8'] == 1
                                                else 'S9' if row['C9'] == 1
                                                else 'S10' if row['C10'] == 1
                                                else 'ITI', axis=1)
data_s1['Size'] = data_s1['Size'].replace(999, pd.NA)



# data_s1['US_expect'] = data_s1['US_expect'].replace(999, pd.NA)
# Here if I translated 999 into np.nan, I am having problem after in the smapling, so I am replacing instead with 0.00
data_s1['US_expect'] = data_s1['US_expect'].replace(999, 0)

# Create a stimulus_phy column.
data_s1['stimulus_phy'] = data_s1['stimulus'].replace({"CS+": "S7"})
# Remove practice trials and ITI trials.
data_s1 = data_s1[(data_s1['block number'] != 2) & (data_s1['stimulus'] != "ITI")]



# Create CS identifier column and Physical size column  .
data_s1['CStrials'] = np.where(data_s1['stimulus'] == 'CS+', 1, 0).astype(int)
data_s1['CS_phy'] =  stimulus_size[6] 

# Rename the Size column to Per_size and select the desired columns.
data_s1.rename(columns={'Size': 'Per_size'}, inplace=True)

selected_cols = ['participant', 'block number', 'trial number', 'US','Startle_Circle','Startle_ITI', 'Per_size', 'US_expect', 'Total_trial_nr', 'stimulus', 'stimulus_phy', 'CStrials','CS_phy']
data_s1 = data_s1[selected_cols]




stimulus_levels = ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"]
stimulus_size_mapping = dict(zip(stimulus_levels, stimulus_size[3:10]))
data_s1['Phy_size'] = data_s1['stimulus'].map(stimulus_size_mapping).astype(float)


data_s1['stimulus'] = pd.Categorical(data_s1['stimulus'], categories=stimulus_level_1, ordered=True)
stimulus_phy_levels = [f'S{i}' for i in range(4, 11)]
data_s1['stimulus_phy'] = pd.Categorical(data_s1['stimulus_phy'], categories=stimulus_phy_levels, ordered=True)


# Create a trial column.
data_s1['trials'] = data_s1.groupby('participant').cumcount() + 1
# data_s1.to_csv("../../vars/r/draft/data_s1_first_py.csv")




# ### 1.2 Long-wide format (for JAGS in R - PYMC in Python)
# 


variable_list_s1 = {
    'y': 'US_expect',
    'Sphy': 'Phy_size',
    'Sper': 'Per_size',
    'CSphy': 'CS_phy',
    'CSindicator1': 'CStrials',
    'CSindicator2': 'CStrials',
    'shock': 'US'
}
jags_input_s1_pre = {}
# Convert the data from long to wide format.
for key, value in variable_list_s1.items():
    df_wide = data_s1.pivot(index='participant', columns='trials', values=value).reset_index()
     # Reorder the rows by participant.
    df_wide = df_wide.set_index('participant').loc[natsorted(df_wide['participant'])].reset_index()
    df_wide = df_wide.drop(columns=['participant'])
    jags_input_s1_pre[key] = df_wide


# Set all values in the CSindicator2 column from trial 15 to 188 to 0.
jags_input_s1_pre['CSindicator2'].iloc[:, 14:188] = 0  




# ### 1.3 Compute distance to CS
# 


# This code chunk has the following purposes:
# 
# Create a CS index by summing the values of the CSindicator1 column and calculating the sum of the CSindicator1 values up to the current column. Extracts CS perception by selecting the Per_size column for rows where stimulus is “CS+”, adding a trials column, and converting the data to wide format. The CS_per_s1 data is reordered by participant and the first column is removed.
# 
# Extracts CS perception by selecting the Per_size column for rows where stimulus is “CS+”, adding a trials column, and converting the data to wide format. The CS_per_s1 data is reordered by participant and the first column is removed.
# 
# Computes the moving average for CS perception by looping through the rows and columns of the CS_per_s1 data and calculating the sum of the CS_per_s1 values up to the current column, excluding NA values.
# 
# Creates empty matrices to store perceptual and physical distance data.
# 
# Computes the perceptual and physical distances by looping through the rows and columns of the data and calculating the absolute value of the difference between the perceptual size/physical size and the CS perception/stimulus size, respectively.
# 
# Merges the distance data with the data_s1 data by participant and trials.


# Create a CS index.
CS_index = np.zeros_like(jags_input_s1_pre['CSindicator1'])


# cs_index 3btswi mtl 3dad l kl participant 3l 3wemid tb3o, iza kan CSindicator1 == 1 3btzid, iza la bb2a nfs l r2m
# Loop through each row and column of the CSindicator1 column.
for i in range(CS_index.shape[0]):
    for j in range(CS_index.shape[1]):
        # Calculate the sum of the CSindicator1 values up to the current column.
        CS_index[i, j] = jags_input_s1_pre['CSindicator1'].iloc[i, :j+1].sum()
# jags_input_s1_pre['CS_index'] = pd.DataFrame(CS_index, index=jags_input_s1_pre['CSindicator1'].index, columns=jags_input_s1_pre['CSindicator1'].columns)
CS_index_df = pd.DataFrame(CS_index, index=jags_input_s1_pre['CSindicator1'].index, columns=jags_input_s1_pre['CSindicator1'].columns)
jags_input_s1_pre['CS_index']  = CS_index_df








# Extract CS perception.
#CS_PER_S1: filter rows of data_s1 where stimulus = cs+ and take only two columns: participant and per_size , shape is random num of rows and two columns
CS_per_s1 = data_s1[data_s1['stimulus'] == 'CS+'][['participant', 'Per_size']].copy()
# adding trials column which is a cummulative based on the participant
CS_per_s1['trials'] = CS_per_s1.groupby('participant').cumcount() + 1
# convert data from long to wide format
CS_per_s1 = CS_per_s1.sort_values(by=['participant', 'trials'])
CS_per_s1_wide = CS_per_s1.pivot(index='participant', columns='trials', values='Per_size').reset_index()
# Now cs_per_s1_wide having each participant in a row and the columns are the values of per_size for this participant but per_size might be NA
# CS_per_s1.drop(columns=CS_per_s1.columns[0], inplace=True)

# Convert all columns to numeric, errors='coerce' will convert non-convertible values to NaN
CS_per_s1 = CS_per_s1.apply(pd.to_numeric, errors='coerce')
CS_per_s1[(CS_per_s1['participant'] == 2) &  (CS_per_s1['trials'] == 4)]
CS_per_s1_wide.set_index('participant', inplace=True)

# Compute moving average for CS perception.
CS_per_updatemean_s1 = pd.DataFrame(index=CS_per_s1_wide.index, columns=CS_per_s1_wide.columns)

for i in range(len(CS_per_s1_wide)):
    for j in range(1,len(CS_per_s1_wide.columns) + 1):
        # Calculate the sum of the CS_per_s1 values up to the current column, excluding NA values.
        CS_per_updatemean_s1.iloc[i,j - 1] = CS_per_s1_wide.iloc[i, :j].dropna().sum() / (j)

# CS per updatemean_s1 is checked compared to R result
# CS_per_updatemean_s1.to_csv(f"{root_folder}vars/r/draft/cs_per_py.csv")



# Create empty matrices for perceptual and physical distance data
d_per = np.zeros((40, 188))
d_phy = np.zeros((40, 188))
d_list_s1 = {'d_per': d_per, 'd_phy': d_phy}

for i in range(1,41):
    for j in range(1,189):
        cs_idx =  jags_input_s1_pre['CS_index'].iloc[i -1, j-1 ]
        s_per = jags_input_s1_pre['Sper'].iloc[i - 1,j - 1]
        if pd.isna(s_per):
            d_per[i-1,j-1] = np.nan
        else: 
            d_per[i- 1,j- 1] = round(abs( s_per -  CS_per_updatemean_s1.iloc[i-1, cs_idx - 1]), 2)

        s_phy = jags_input_s1_pre['Sphy'].iloc[i - 1,j - 1]
        d_phy[i-1,j-1] = np.round(abs(s_phy - stimulus_size[6]),2)

d_list_s1['d_per'] = d_per
d_list_s1['d_phy'] = d_phy


#d_phy, d_per is correct and compared with R
# pd.DataFrame(d_per).to_csv(f"{root_folder}vars/r/draft/d_per.csv")
# pd.DataFrame(d_phy).to_csv(f"{root_folder}vars/r/draft/d_phy.csv")


# dr = pd.melt(d_list_s1['d_per'], var_name=["participant", "trials"], value_name="dper").rename(columns={"participant": "Var1", "trials": "Var2"})
reshaped = {}
for key in ['d_per', 'd_phy']:
    ar = d_list_s1[key].flatten()
    reshaped[key] = pd.DataFrame({
        "participant": np.repeat(range(1, d_per.shape[0] + 1), d_per.shape[1]),
        "trials": np.tile(range(1, d_per.shape[1] + 1), d_per.shape[0]),
        key: ar,
    })
    # reshaped[key]['participant'] = reshaped[key]['participant'].astype(str)

    reshaped[key] = reshaped[key].sort_values(by=["trials", "participant"]).reset_index(drop=True)
    data_s1 = data_s1.merge(reshaped[key], on=['participant', 'trials'])

# data_s1.columns
data_s1 = data_s1[['participant', 'trials', "d_phy" , "d_per", "block number", "trial number", "US", "Startle_Circle", "Startle_ITI", "Per_size", "US_expect", "Total_trial_nr", "stimulus", "stimulus_phy", "CStrials", "CS_phy", "Phy_size"]]
# data_s1.to_csv(f"{root_folder}vars/r/draft/ds1_py.csv")




# ### 1.4 JAGS (PYMC) input file :
# 



def data_input_s1(L, indicator=None):
    data = {
        'Nparticipants': jags_input_s1_pre['y'].shape[0],
        'Ntrials': jags_input_s1_pre['y'].shape[1],
        'Nactrials': 14,
        'd_per': [d_list_s1['d_per'][:, 14:188], d_list_s1['d_per']][L-1],
        'd_phy': [d_list_s1['d_phy'][:, 14:188], d_list_s1['d_phy']][L-1],
        'y': np.array([jags_input_s1_pre['y'].iloc[:, 14:188], jags_input_s1_pre['y']][L-1])
    }
    
    if L == 2:
        additional_data = {
            'r': np.array(jags_input_s1_pre['shock']),
            'k': np.array([jags_input_s1_pre['CSindicator1'], jags_input_s1_pre['CSindicator2']][indicator-1])
        }
        data.update(additional_data)
    
    return data

#input data without learning trials
Data1_JAGSinput_G = data_input_s1(1)
#input data with an assumption of non-continuous learning 
Data1_JAGSinput_LG = data_input_s1(2, 2)
#input data with an assumption of continuous learning 
Data1_JAGSinput_CLG = data_input_s1(2, 1)









def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


# Data1_JAGSinput_G has been compared with R and it is correct
# Data1_JAGSinput_LG has been compared with R and it is correct
# Data1_JAGSinput_CLG has been compared with R and it is correct
# data_s1 has been compared with R and it is correct

# Save as pickles
save_data_as_pickle(Data1_JAGSinput_G, f'{root_folder}Data/res_py/Data1_JAGSinput_G.pkl')
save_data_as_pickle(Data1_JAGSinput_LG, f'{root_folder}Data/res_py/Data1_JAGSinput_LG.pkl')
save_data_as_pickle(Data1_JAGSinput_CLG, f'{root_folder}Data/res_py/Data1_JAGSinput_CLG.pkl')
save_data_as_pickle(data_s1, f'{root_folder}Data/res_py/Data_s1.pkl')




# ## 2. Dataset 2: Differential conditioning
# 


# ### 2.1 Pre-process
# 


# Create a data list for data loading
participants_2 = list(range(41, 81))
data_s2_list = [f"{root_folder}Data/Experiment_2/{p}/{p}_results.txt" for p in participants_2]
data_s2 = pd.concat([pd.read_csv(file, sep="\t") for file in data_s2_list], keys=range(1, 41), names=['participant']).reset_index()


# Load data to 'data_s2' and process data

conditions = [
    ((data_s2['C1'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C10'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C2'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C9'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C3'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C8'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C4'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C7'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C5'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C6'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C6'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C5'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C7'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C4'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C8'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C3'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C9'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C2'] == 1) & (data_s2['group'] == 2)),
    ((data_s2['C10'] == 1) & (data_s2['group'] == 1)) | ((data_s2['C1'] == 1) & (data_s2['group'] == 2))
]
choices = ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"]
  # Create stimulus column ; Change 999 to NA
data_s2['stimulus'] = np.select(conditions, choices, default='ITI')
data_s2['Size'] = data_s2['Size'].replace(999, np.nan)

# data_s2['US_expect'] = data_s2['US_expect'].replace(999, np.nan)
# Here if I translated 999 into np.nan, I am having problem after in the smapling, so I am replacing instead with 0.00
data_s2['US_expect'] = data_s2['US_expect'].replace(999, 0)

conditions = [
    (data_s2['C1'] == 1),
    (data_s2['C2'] == 1),
    (data_s2['C3'] == 1),
    (data_s2['C4'] == 1),
    (data_s2['C5'] == 1),
    (data_s2['C6'] == 1),
    (data_s2['C7'] == 1),
    (data_s2['C8'] == 1),
    (data_s2['C9'] == 1),
]
choices = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
data_s2['stimulus_phy'] = np.select(conditions, choices, default='S10')

data_s2['CSptrials'] = np.where(data_s2['stimulus'] == "CS+", 1,0)
data_s2['CSmtrials'] = np.where(data_s2['stimulus'] == 'CS-', 1, 0)
# Remove practice trials and ITI trials
data_s2 = data_s2[(data_s2['block number'] != 2) & (data_s2['stimulus'] != 'ITI')]
data_s2['USp'] = np.where((data_s2['CSptrials'] == 1) & (data_s2['US'] == 1), 1, 0)
data_s2['USm'] = np.where((data_s2['CSmtrials'] == 1) & (data_s2['US'] == 0), -1, 0)
data_s2['CS_phy_p'] = np.where(data_s2['group'] == 1, stimulus_size[0], stimulus_size[9])
data_s2['CS_phy_m'] = np.where(data_s2['group'] == 1, stimulus_size[9], stimulus_size[0])
data_s2 = data_s2.rename(columns={'Size': 'Per_size'})

# Create a new column called 'Phy_size' that maps the stimulus_phy column to a numeric value based on the values in 'stimulus_size'
stimulus_phy_map = {f'S{i}': stimulus_size[i-1] for i in range(1, 11)}
data_s2['Phy_size'] = data_s2['stimulus_phy'].map(stimulus_phy_map)
# Convert the 'stimulus' and 'stimulus_phy' columns to factors
data_s2['stimulus'] = pd.Categorical(data_s2['stimulus'], categories=stimulus_level_2)
data_s2['stimulus_phy'] = pd.Categorical(data_s2['stimulus_phy'], categories=[f'S{i}' for i in range(1, 11)])
# Create a new column called 'trials' that is a sequence of numbers for each participant
data_s2['trials'] = data_s2.groupby('participant').cumcount() + 1

selected_cols=  ["participant", "block number", "trial number", "US", "Startle_Circle", "Startle_ITI","Per_size", "US_expect", "Total_trial_nr", "group", "stimulus", "stimulus_phy", "CSptrials", "CSmtrials", "USp", "USm","CS_phy_p", "CS_phy_m", 'Phy_size', 'trials']

data_s2 = data_s2[selected_cols]
# data_s2.to_csv(f"{root_folder}vars/r/draft/ds2py.csv")


# ### 2.2 Long-wide format (for JAGS - PYMC)
# 



# Create a list of variables to extract from the long format data
variable_list_s2 = {
    'y': 'US_expect', 
    'Sphy': 'Phy_size', 
    'Sper': 'Per_size', 
    'CSphy_p': 'CS_phy_p', 
    'CSphy_m': 'CS_phy_m',
    'CSindicator1_p': 'CSptrials', 
    'CSindicator2_p': 'CSptrials', 
    'CSindicator1_m': 'CSmtrials', 
    'CSindicator2_m': 'CSmtrials', 
    'US_p': 'USp', 
    'US_m': 'USm'
}


# Convert the data from long to wide format
jags_input_s2_pre = {}
for key, value in variable_list_s2.items():
    wide_df = data_s2.pivot(index='participant', columns='trials', values=value)
    # Reorder the rows by participant
    wide_df.sort_index(inplace=True)
    jags_input_s2_pre[key] = wide_df

# Set all values in the CSindicator2_p column from trial 25 to 180 to 0
jags_input_s2_pre['CSindicator2_p'].iloc[:, 24:180] = 0  
# Set all values in the CSindicator2_m column from trial 25 to 180 to 0
jags_input_s2_pre['CSindicator2_m'].iloc[:, 24:180] = 0  


# jags_input_s2_pre WAS COMPARED IN THE CURRENT STAGE AND IT IS MATCHED
# for key, df in jags_input_s2_pre.items():
#     df.to_csv(f"{root_folder}vars/r/draft/jags_py/jags_input_s2_pre_{key}_py.csv", index=False)



# ### 2.3 Compute distance to CS
# 



# Create CS index
# Initialize empty matrices for the CSp_index and CSm_index
jags_input_s2_pre['CSp_index'] = np.zeros_like(jags_input_s2_pre['CSindicator1_p'])
jags_input_s2_pre['CSm_index'] = np.zeros_like(jags_input_s2_pre['CSindicator1_m'])
# Loop through each row and column of the CSindicator1_p matrix
for i in range(jags_input_s2_pre['CSindicator1_p'].shape[0]):
    for j in range(jags_input_s2_pre['CSindicator1_p'].shape[1]):
        # For each element in the matrix, set the value to the sum of the values in the first j columns of the same row
        jags_input_s2_pre['CSp_index'][i, j] = np.sum(jags_input_s2_pre['CSindicator1_p'].iloc[i, :j+1])
for i in range(jags_input_s2_pre['CSindicator1_m'].shape[0]):
    for j in range(jags_input_s2_pre['CSindicator1_m'].shape[1]):
        # For each element in the matrix, set the value to the sum of the values in the first j columns of the same row
        jags_input_s2_pre['CSm_index'][i, j] = np.sum(jags_input_s2_pre['CSindicator1_m'].iloc[i, :j+1])
jags_input_s2_pre['CSp_index'] = pd.DataFrame(jags_input_s2_pre['CSp_index'])
jags_input_s2_pre['CSm_index'] = pd.DataFrame(jags_input_s2_pre['CSm_index'])


# For all elements in the CSp_index matrix that are equal to 0, set their value to 1
jags_input_s2_pre['CSp_index'] = jags_input_s2_pre['CSp_index'].replace(0, 1)
# For all elements in the CSm_index matrix that are equal to 0, set their value to 1
jags_input_s2_pre['CSm_index'] = jags_input_s2_pre['CSm_index'].replace(0, 1)






# Correct comparing to R results
# jags_input_s2_pre['CSm_index'].to_csv(f"{root_folder}vars/r/draft/csm_idx_py.csv")
# jags_input_s2_pre['CSp_index'].to_csv(f"{root_folder}vars/r/draft/csp_idx_py.csv")






# Define the stimuli to process
stimuli = [ "CS+","CS-"]
# stimuli = [ "CS-"]
# Create a list to store results
CS_per_s2_list = []
# Create a list of data frames containing the Per_size column for the "CS+" and "CS-" stimuli
for stimulus in stimuli:
    # Filter the data for the current stimulus and select required columns
    CS_per = data_s2[data_s2['stimulus'] == stimulus][['participant', 'Per_size']]
    # Add a 'trials' column that numbers each row within each participant
    CS_per['trials'] = CS_per.groupby('participant').cumcount() + 1
    CS_per = CS_per[['participant', 'trials', 'Per_size']]
    # Pivot the data to wide format
    CS_per = CS_per.pivot(index='participant', columns='trials', values='Per_size')
    CS_per.reset_index( inplace=True)
    
    # Initialize an empty matrix for storing the moving average values
    CS_per_updatemean = pd.DataFrame(index=CS_per.index, columns=CS_per.columns)
    
    # Loop through each row and column of the CS_per DataFrame to compute the moving averages
    for index, row in CS_per.iterrows():
        for col in CS_per.columns:
            # For each element in the matrix, set the value to the mean of the values in the first j columns of the corresponding row
            values = row.loc[1:col].replace(np.nan, 0)  # Select values up to the current column
            mean= values.mean(skipna=False)  # Compute mean, skipping NaNs
            CS_per_updatemean.at[index, col] = round(mean,4)   
    CS_per_updatemean = pd.DataFrame(CS_per_updatemean)
    CS_per_updatemean.set_index("participant", inplace=True)
    # Store the DataFrame of moving average values in the list
    CS_per_s2_list.append(CS_per_updatemean)

# CS_per_s2_list is correct using R comparison



# d_list_s2
# Initialize a list of four empty matrices for storing the distance values
d_per_p = np.zeros((40, 180))
d_per_m = np.zeros((40, 180))
d_phy_p = np.zeros((40, 180))
d_phy_m = np.zeros((40, 180))

# This code loops through two nested for loops to calculate differences between
# two variables (Sper, Sphy) and their corresponding counterparts (CS_per_s2_list, 
# CSphy_p, CSphy_m). The differences are calculated for each participant (1:40) 
# and each trial (1:180) and are stored in a data frame (d_list_s2).
for i in range(40): #participants
    for j in range(180): #trials
        # Calculate difference between Sper and corresponding CS_per value for each trial and participant
        # a=jags_input_s2_pre['Sper'].iloc[i, j]
        # idx2 = int(jags_input_s2_pre['CSp_index'].iloc[i, j]) - 1
        # b= CS_per_s2_list[0].iloc[i, idx2]
        
        d_per_p[i, j] = round(abs(jags_input_s2_pre['Sper'].iloc[i, j] - CS_per_s2_list[0].iloc[i, int(jags_input_s2_pre['CSp_index'].iloc[i, j] - 1)]), 2)
        d_per_m[i, j] = round(abs(jags_input_s2_pre['Sper'].iloc[i, j] - CS_per_s2_list[1].iloc[i, int(jags_input_s2_pre['CSm_index'].iloc[i, j]) - 1]), 2)
        
        
        # Calculate difference between Sphy and CSphy values for each trial and participant
        d_phy_p[i, j] = round(abs(jags_input_s2_pre['Sphy'].iloc[i, j] - jags_input_s2_pre['CSphy_p'].iloc[i, j]), 2)
        d_phy_m[i, j] = round(abs(jags_input_s2_pre['Sphy'].iloc[i, j] - jags_input_s2_pre['CSphy_m'].iloc[i, j]), 2)




# # Convert matrices to DataFrames for merging
d_per_p_df = pd.DataFrame(d_per_p, columns=[f'trial_{i+1}' for i in range(180)])
d_per_m_df = pd.DataFrame(d_per_m, columns=[f'trial_{i+1}' for i in range(180)])
d_phy_p_df = pd.DataFrame(d_phy_p, columns=[f'trial_{i+1}' for i in range(180)])
d_phy_m_df = pd.DataFrame(d_phy_m, columns=[f'trial_{i+1}' for i in range(180)])


d_per_p_df.index = d_per_p_df.index + 1
d_per_m_df.index = d_per_m_df.index + 1
d_phy_p_df.index = d_phy_p_df.index + 1
d_phy_m_df.index = d_phy_m_df.index + 1


def geValue_per_p(row):
    participant = row['participant']
    trial = row["trials"]
    column_name = f'trial_{trial}'
    return d_per_p_df.at[participant, column_name]

def geValue_per_m(row):
    participant = row['participant']
    trial = row["trials"]
    column_name = f'trial_{trial}'
    return d_per_m_df.at[participant, column_name]


def geValue_phy_p(row):
    participant = row['participant']
    trial = row["trials"]
    column_name = f'trial_{trial}'
    return d_phy_p_df.at[participant, column_name]


def geValue_phy_m(row):
    participant = row['participant']
    trial = row["trials"]
    column_name = f'trial_{trial}'
    return d_phy_m_df.at[participant, column_name]

data_s2["d_per_p"] = data_s2.apply(geValue_per_p, axis=1)
data_s2["d_per_m"] = data_s2.apply(geValue_per_m, axis=1)
data_s2["d_phy_p"] = data_s2.apply(geValue_phy_p, axis=1)
data_s2["d_phy_m"] = data_s2.apply(geValue_phy_m, axis=1)

# data_s2 = data_s2[["participant", "trials", "d_phy_m", "d_phy_p", "d_per_m", "d_per_p", "version", "timestamp", "block number", "trial number", "lost msec", "free VRAM", "trial contents", "US", "Startle_Circle", "Startle_ITI", "C1", "C2", "C3", "C5", "C4", "C6", "C7", "C8", "C9", "C10", "No_Startle", "Generalization_Block", "No_US", "Per_size", "US_expect", "Resp_Rev", "US_Intensity", "Total_trial_nr", "group", "stimulus", "stimulus_phy", "CSptrials", "CSmtrials", "USp", "USm", "CS_phy_p", "CS_phy_m", "Phy_size"]]
data_s2_cols = ["participant", "trials", "d_phy_m", "d_phy_p", "d_per_m", "d_per_p", "block number", "trial number",  "US", "Startle_Circle", "Startle_ITI", "Per_size","US_expect","Total_trial_nr", "group", "stimulus", "stimulus_phy","CSptrials","CSmtrials", "USp", "USm","CS_phy_p","CS_phy_m","Phy_size"]
data_s2 = data_s2[data_s2_cols]

d_list_s2 = [d_per_p_df, d_per_m_df, d_phy_p_df, d_phy_m_df]


# ALL CORRECT COMPARED  TO R 
# d_per_p_df.to_csv(f"{root_folder}vars/r/draft/d_list_py/l_d_per_p_py.csv")
# d_per_m_df.to_csv(f"{root_folder}vars/r/draft/d_list_py/l_d_per_m_py.csv")
# d_phy_p_df.to_csv(f"{root_folder}vars/r/draft/d_list_py/l_d_phy_p_py.csv")
# d_phy_m_df.to_csv(f"{root_folder}vars/r/draft/d_list_py/l_d_phy_m_py.csv")




# ### 2.4 JAGS(PYMC) input file
# 



def data_input_s2(L, indicator=None):
    # Initialize data dictionary with fixed structure
    data = {
        'Nparticipants': jags_input_s2_pre['y'].shape[0],
        'Ntrials': jags_input_s2_pre['y'].shape[1],
        'Nactrials': 24,
        'd_p_per': d_list_s2[0].iloc[:, 24:180] if L == 1 else d_list_s2[0],
        'd_m_per': d_list_s2[1].iloc[:, 24:180] if L == 1 else d_list_s2[1],
        'd_p_phy': d_list_s2[2].iloc[:, 24:180] if L == 1 else d_list_s2[2],
        'd_m_phy': d_list_s2[3].iloc[:, 24:180] if L == 1 else d_list_s2[3],
        'y': jags_input_s2_pre['y'].iloc[:, 24:180] if L == 1 else jags_input_s2_pre['y'],
        # 'y2': np.array([jags_input_s2_pre['y'][:, 24:180], jags_input_s2_pre['y']][L - 1])
    }

    # Add conditional data based on the value of L and indicator
    if L == 2:
        additional_data = {
            'r_plus': jags_input_s2_pre['US_p'],
            'r_minus': jags_input_s2_pre['US_m'],
            'k_plus': jags_input_s2_pre['CSindicator1_p' if indicator == 1 else 'CSindicator2_p'],
            'k_minus': jags_input_s2_pre['CSindicator1_m' if indicator == 1 else 'CSindicator2_m']
        }
        data.update(additional_data)

    return data



# Usage of the function with the specified arguments
Data2_JAGSinput_G = data_input_s2(1)
Data2_JAGSinput_LG = data_input_s2(2, 2)
Data2_JAGSinput_CLG = data_input_s2(2, 1)



save_data_as_pickle(Data2_JAGSinput_G, f"{root_folder}Data/res_py/Data2_JAGSinput_G.pkl")
save_data_as_pickle(Data2_JAGSinput_LG, f"{root_folder}Data/res_py/Data2_JAGSinput_LG.pkl")
save_data_as_pickle(Data2_JAGSinput_CLG, f"{root_folder}Data/res_py/Data2_JAGSinput_CLG.pkl")
save_data_as_pickle(data_s2, f"{root_folder}Data/res_py/Data_s2.pkl")






#Data2_JAGSinput_G, Data2_JAGSinput_LG, Data2_JAGSinput_CLG, data_s2 are checked and compared with R and they are correct
# data_s2.to_csv(f"{root_folder}vars/r/draft/ds2_py.csv")

# keys = ['d_p_per', 'd_m_per', 'd_p_phy', 'd_m_phy', 'y',]
# additional_keys = [ 'r_plus', 'r_minus', 'k_plus', 'k_minus']

# _dfs = {
#     "G": {"items": Data2_JAGSinput_G, "keys": keys }, 
#     "LG": {"items": Data2_JAGSinput_LG, "keys": keys  + additional_keys}, 
#     "CLG": {"items": Data2_JAGSinput_CLG, "keys": keys  + additional_keys}, 
# }

# for frm, _v in _dfs.items():
#     for k in _v['keys']:
#         pd.DataFrame(_v["items"][k]).to_csv(f"{root_folder}vars/r/draft/jags2/{k}_{frm}_py.csv")


# ### Save data files as csv:


data_s1.to_csv(f"{root_folder}Data/res_py/data_s1_py.csv")
data_s2.to_csv(f"{root_folder}Data/res_py/data_s2_py.csv")

