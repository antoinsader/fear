import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from random import seed, sample


root_folder ="../../"
plts_folder = root_folder + "Plots/py/6_gp_allocation/"


def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
    
def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
            

gp_allocs = load_data_from_pickle(f"{root_folder}Data/res_py/2_an_gp_allocs.pkl")




data =[
    load_data_from_pickle(f"{root_folder}Data/res_py/Data_s1.pkl"),
    load_data_from_pickle(f"{root_folder}Data/res_py/Data_s2.pkl"),
]


y_samples = [
    load_data_from_pickle(f"{root_folder}Data/res_py/result_study1_y_pre.pkl"),
    load_data_from_pickle(f"{root_folder}Data/res_py/result_study2_y_pre.pkl"),
]


Ntrials_1 = 188
Ntrials_2 = 180
Nactrials_1 = 14
Nactrials_2 = 24
experiments = ["Exp.1: Simple conditioning" , "Exp.2: "]
group_nm = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers", "Unknown"]
group = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers"]
color_gp = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]




def create_gp_allo_plots(gp_allocation, experiments, group_nm, color_gp):
    for i, exp in enumerate(experiments):
        data = gp_allocation[i][0].copy()  # First element is the DataFrame
        data["value"] = data["value"].astype(str)
        g = sns.FacetGrid(
            data=data,
            col="value",
            col_wrap=1,
            height=4,
            sharex=True,
            sharey=True,
            aspect=3
        )
        g.map_dataframe(
            sns.barplot,
            x="Var2",
            y="prop",
            hue="value",
            palette=color_gp,
            dodge=False
        )

        for ax in g.axes.flat:
            ax.axhline(y=0.75, color="black", linestyle="dashed", linewidth=1)  # Add dashed horizontal line
            ax.set_ylim(0, 1)  # Set consistent y-axis limits
        facet_labels = dict(zip(["4.0", "3.0", "2.0", "1.0"], group_nm[::-1]))
        for ax, value in zip(g.axes.flat, data["value"].unique()):
            ax.set_title(facet_labels[value])
        g.set_axis_labels("Participant", "Sample proportion")
        g.fig.suptitle(exp, fontsize=16, y=1.02)
        g.add_legend(title="Groups", loc="upper right", labels=group_nm[1:4], frameon=False)
        g.tight_layout()

        g.savefig(f"{plts_folder}gp_allocation_{i}.png", dpi=300)
        
        plt.show()

plots = create_gp_allo_plots(gp_allocs, experiments, group_nm, color_gp)


def calculate_quantiles(df, group_cols, target_col, sim_col):
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    stats = df.groupby(group_cols).agg(
        **{f'q{int(q * 100)}': (target_col, lambda x: np.quantile(x.dropna(), q)) for q in quantiles},
        mean=(target_col, lambda x: x.dropna().mean()),
        s_q2_5=(sim_col, lambda x: np.quantile(x, 0.025)),
        s_q10=(sim_col, lambda x: np.quantile(x, 0.1)),
        s_q30=(sim_col, lambda x: np.quantile(x, 0.3)),
        s_q50=(sim_col, lambda x: np.quantile(x, 0.5)),
        s_q70=(sim_col, lambda x: np.quantile(x, 0.7)),
        s_q90=(sim_col, lambda x: np.quantile(x, 0.9)),
        s_q97_5=(sim_col, lambda x: np.quantile(x, 0.975)),
        s_mean=(sim_col, lambda x: x.mean()),
    ).reset_index()
    return stats



pre_data =[]

for x in range(2):
    y_pre = y_samples[x]
    start_idx = (Nactrials_1, Nactrials_2)[x]
    end_idx = (Ntrials_1, Ntrials_2)[x]
    sample_numbers = sample(range(0, 10000), 5000)
    
    reshaped_data = y_pre[np.ix_(sample_numbers, range(y_pre.shape[1]), range(start_idx, end_idx))]
    samples, participants, trials = reshaped_data.shape
    
    d = pd.DataFrame({
        "samples": np.repeat(np.arange(samples), participants * trials),
        "participant": np.tile(np.repeat(np.arange(participants), trials), samples),
        "trials": np.tile(np.arange(trials), samples * participants),
        "sim_y": reshaped_data.ravel()  # Flatten the array for the 'value' column
    })
    d['trials'] += (Nactrials_1, Nactrials_2)[x]

    
    d['samples'] += 1
    d['trials'] += 1
    d['participant'] += 1

    d['participant'] = d['participant'].astype(str)

    data_filtered = data[x][data[x]['trials'] > (Nactrials_1, Nactrials_2)[x]]
    data_filtered['participant'] = data_filtered['participant'].astype(str)
    
    
    
    d = d.merge(data_filtered[['participant', 'trials', 'US_expect']], 
                    on=['participant', 'trials'], how='left').rename(columns={'US_expect': 'y'})
    
    d = d.merge(data_filtered[['participant', 'trials', 'stimulus']], 
                on=['participant', 'trials'], how='left')
    
    d = d.merge(gp_allocs[x][1], on='participant', how='left').rename(columns={'group': 'category'})


    # Group-level quantiles
    d_filtered_forgroup = d.loc[
        (d['samples'] == 1) & (d['participant'] == '1')
    ]
    
    d_group = calculate_quantiles(d_filtered_forgroup, ['stimulus'], 'y', 'sim_y')

    # Category-level quantiles
    d_category = calculate_quantiles(d, ['stimulus', 'category'], 'y', 'sim_y')
    d_category = d_category.sample(n=1)

    # Individual-level quantiles
    d_indi = calculate_quantiles(d, ['stimulus', 'participant'], 'y', 'sim_y')
    d_indi = d_indi.sample(n=1)
    
   

    pre_data.append([d_group, d_category, d_indi])
    


    



save_data_as_pickle(pre_data, f"{root_folder}Data/res_py/2_an_predata.pkl")


