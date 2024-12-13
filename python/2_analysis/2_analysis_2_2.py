# Here we will try to organize the results for data optimization after 

import pickle
import numpy as np

root_folder ="../../"


def save_data_as_pickle(filename, data ):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
    

result = [
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study1_CLG2D_t.pkl"),
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study2_CLG2D_t.pkl")
]





save_data_as_pickle(f"{root_folder}Data/res_py/result_study1_y_pre.pkl", np.array(result[0]['sims_list']['y_pre']))
save_data_as_pickle(f"{root_folder}Data/res_py/result_study2_y_pre.pkl", np.array(result[1]['sims_list']['y_pre']))



save_data_as_pickle(f"{root_folder}Data/res_py/result_study1_gp.pkl", np.array(result[0]['sims_list']['gp']))
save_data_as_pickle(f"{root_folder}Data/res_py/result_study2_gp.pkl", np.array(result[1]['sims_list']['gp']))

