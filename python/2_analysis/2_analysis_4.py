import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt



root_folder = "../../"
plots_posterior_folder= root_folder + "Plots/py/5_posterior_predictive/"

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
        

result = [
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study1_CLG2D_t.pkl"),
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study2_CLG2D_t.pkl")
]



gp_samples = [
    result[0]['sims_list']['gp'],
    result[1]['sims_list']['gp'],
]





Nparticipant = 40
Ntrials_1 = 188
Ntrials_2 = 180
Nactrials_1 = 14
Nactrials_2 = 24



experiments = ["Exp.1: Simple conditioning" , "Exp.2: "]
color = ["#56B4E9", "yellow", "#CC79A7", "#009E73"]
group_nm = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers", "Unknown"]
colors = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]
stimulus_level_study1 = ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"]
stimulus_level_study2 = ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"]
group = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers"]
color_gp = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]





def process_gp_samples(gp_samples):
    results = []
    
    for sample in gp_samples:
        sample = pd.DataFrame(sample)
        df = pd.melt(sample.reset_index(), id_vars=["index"], var_name="Var2", value_name="value")
        df["Var2"] += 1
        
        df_grouped = df.groupby(["Var2", "value"]).size().reset_index(name="n")
        df_grouped["prop"] = df_grouped["n"] / 10000  # Equivalent to `mutate(prop = n / 10000)`
        
        df2 = df_grouped.loc[df_grouped.groupby("Var2")["n"].idxmax()].copy()
        
        def assign_group(row):
            if row["prop"] > 0.75:
                if row["value"] == 1:
                    return "Non-Learners"
                elif row["value"] == 2:
                    return "Overgeneralizers"
                elif row["value"] == 3:
                    return "Physical Generalizers"
                elif row["value"] == 4:
                    return "Perceptual Generalizers"
            return "Unknown"

        def assign_group2(row):
            if row["value"] == 1:
                return "Non-Learners"
            elif row["value"] == 2:
                return "Overgeneralizers"
            elif row["value"] == 3:
                return "Physical Generalizers"
            elif row["value"] == 4:
                return "Perceptual Generalizers"

        df2["group"] = df2.apply(assign_group, axis=1)
        df2["group2"] = df2.apply(assign_group2, axis=1)
        
        # Rename Var2 to participant
        df2.rename(columns={"Var2": "participant"}, inplace=True)
        df2["participant"] = df2["participant"].astype(str)
        df2["group"] = df2["group"].astype(str)
        
        # Add the ratio and ratio2 columns to df
        non_learners_ratio = sum(df2["group"] == "Non-Learners") / 40 * 100
        overgeneralizers_ratio = sum(df2["group"] == "Overgeneralizers") / 40 * 100
        physical_ratio = sum(df2["group"] == "Physical Generalizers") / 40 * 100
        perceptual_ratio = sum(df2["group"] == "Perceptual Generalizers") / 40 * 100
        
        non_learners_ratio2 = sum(df2["group2"] == "Non-Learners") / 40 * 100
        overgeneralizers_ratio2 = sum(df2["group2"] == "Overgeneralizers") / 40 * 100
        physical_ratio2 = sum(df2["group2"] == "Physical Generalizers") / 40 * 100
        perceptual_ratio2 = sum(df2["group2"] == "Perceptual Generalizers") / 40 * 100
        
        def calculate_ratio(row):
            if row["value"] == 1:
                return non_learners_ratio
            elif row["value"] == 2:
                return overgeneralizers_ratio
            elif row["value"] == 3:
                return physical_ratio
            elif row["value"] == 4:
                return perceptual_ratio

        def calculate_ratio2(row):
            if row["value"] == 1:
                return non_learners_ratio2
            elif row["value"] == 2:
                return overgeneralizers_ratio2
            elif row["value"] == 3:
                return physical_ratio2
            elif row["value"] == 4:
                return perceptual_ratio2

        df_grouped["ratio"] = df_grouped.apply(calculate_ratio, axis=1)
        df_grouped["ratio2"] = df_grouped.apply(calculate_ratio2, axis=1)
        
        results.append((df_grouped, df2))
    
    return results

gp_allocs = process_gp_samples(gp_samples)
save_data_as_pickle(gp_allocs, f"{root_folder}Data/res_py/2_an_gp_allocs.pkl")


