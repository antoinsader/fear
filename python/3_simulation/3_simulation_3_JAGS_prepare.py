
import pickle
import pandas as pd 
import numpy as np

root_folder= "../../"
data_folder= root_folder + "Data/"


def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
data = load_data_from_pickle(f"{data_folder}res_py/Data_s2.pkl")
data = data[data['participant'] == 1]

data = data.sort_values(by=['participant', 'trials'])

Ngroup1 = 50
Ngroup2 = 50
Ngroup3 = 50
Ngroup4 = 50

# Calculate the total number of participants
Nparticipants = Ngroup1 + Ngroup2 + Ngroup3 + Ngroup4

# Get the number of trials from the data
Ntrials = int(data['trials'].max())

# Set the number of actual trials (excluding warm-up and cool-down trials)
Nactrials = 24

# Define the group names and colors for plots
group_nm = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers", "Unknown"]
color_gp = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00"]



def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)



result = pd.read_csv(f"{data_folder}res_py/result_3_simulation.csv")

q = np.reshape(result['d_per_p'], (Nparticipants, -1), order='F')[:,]
q2= np.reshape(result['d_per_p'], (Nparticipants, -1), order='F')




def PYMCData(L):
    return {
        "Nparticipants": Nparticipants,
        "Ntrials": [Ntrials- Nactrials + 1 , Ntrials   ][L],
        "Nactrials": Nactrials,
         "y": [np.reshape(result.y, (Nparticipants, -1), order='F')[:, Nactrials-1:Ntrials], np.reshape(result.y, (Nparticipants, -1), order='F')][L].ravel(order="f"),
        "d_p_per": [np.reshape(result['d_per_p'], (Nparticipants, -1), order='F')[:, Nactrials-1:Ntrials], np.reshape(result['d_per_p'], (Nparticipants, -1), order='F')][L].ravel(order="f"),
        "d_m_per": [np.reshape(result['d_per_m'], (Nparticipants, -1), order='F')[:, Nactrials-1:Ntrials], np.reshape(result['d_per_m'], (Nparticipants, -1), order='F')][L].ravel(order="f"),
        "d_p_phy": [np.reshape(result['d_phy_p'], (Nparticipants, -1), order='F')[:, Nactrials-1:Ntrials], np.reshape(result['d_phy_p'], (Nparticipants, -1), order='F')][L].ravel(order="f"),
        "d_m_phy": [np.reshape(result['d_phy_m'], (Nparticipants, -1), order='F')[:, Nactrials-1:Ntrials], np.reshape(result['d_phy_m'], (Nparticipants, -1), order='F')][L].ravel(order="f"),
       
    }

save_data_as_pickle(PYMCData(1), f"{data_folder}/res_py/Sim_PYMCinput_CLG.pkl")
save_data_as_pickle(PYMCData(0), f"{data_folder}res_py/Sim_PYMCinput_G.pkl")



