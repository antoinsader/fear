import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.backends.backend_pdf import PdfPages

root_folder= "../../";
plot_folder = root_folder + 'Plots/py/7_parameterEstimation/'
def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
    


stimulus_level_study1 = ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"]
stimulus_level_study2 = ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"]

data =[
    load_data_from_pickle(f"{root_folder}Data/res_py/Data_s1.pkl"),
    load_data_from_pickle(f"{root_folder}Data/res_py/Data_s2.pkl"),
]

data[0]["stim_gr"] = 1
gp_allocs = load_data_from_pickle(f"{root_folder}Data/res_py/2_an_gp_allocs.pkl")


# Example for gp_allocation
gp_allocation = gp_allocs


result = [
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study1_CLG2D_t.pkl"),
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study2_CLG2D_t.pkl")
]



geplot_all=[]
data_ge = []


for i in range(2):
    # Merge group allocation into data[i]
    data[i] = data[i].merge(
        gp_allocation[i][1], on='participant', how='left'
    )
    data[i]['category'] = data[i]['i.group']

    # Set stimulus levels
    data[i]['stimulus'] = pd.Categorical(
        data[i]['stimulus'],
        categories=[stimulus_level_study1, stimulus_level_study2][i]
    )

    # Group and calculate metrics
    data[i]['mean_per_indi'] = data[i].groupby(['stimulus', 'participant'])['Per_size'].transform('mean')
    data[i]['sd_per_indi'] = data[i].groupby(['stimulus', 'participant'])['Per_size'].transform('std')

    data[i]['mean_us_category'] = data[i].groupby(['stimulus', 'category'])['US_expect'].transform('mean')
    data[i]['mean_per_group'] = data[i].groupby(['stimulus_phy', 'category'])['Per_size'].transform('mean')
    data[i]['sd_per_group'] = data[i].groupby(['stimulus_phy', 'category'])['Per_size'].transform('std')

    data[i]['mean_us_group'] = data[i].groupby(['stimulus'])['US_expect'].transform('mean')
    data[i]['mean_us_trial'] = data[i].groupby(['trials', 'category', 'stimulus'])['US_expect'].transform('mean')
    data[i]['mean_us_trial_all'] = data[i].groupby(['trials', 'stimulus'])['US_expect'].transform('mean')

    # Filter unknown category
    data[i] = data[i][data[i]['category'] != "Unknown"]

data[0]['mean_disp_group'] = data[0].groupby(['stimulus', 'category'])['dper'].transform('mean')
data[0]['sd_disp_group'] = data[0].groupby(['stimulus', 'category'])['dper'].transform('std')
data[0]['mean_disp_indi'] = data[0].groupby(['stimulus', 'participant'])['dper'].transform('mean')
data[0]['sd_disp_indi'] = data[0].groupby(['stimulus', 'participant'])['dper'].transform('std')

data[1]['mean_disp_group'] = data[1].groupby(['stimulus', 'category'])['d_per_p'].transform('mean')
data[1]['sd_disp_group'] = data[1].groupby(['stimulus', 'category'])['d_per_p'].transform('std')
data[1]['mean_disp_indi'] = data[1].groupby(['stimulus', 'participant'])['d_per_p'].transform('mean')
data[1]['sd_disp_indi'] = data[1].groupby(['stimulus', 'participant'])['d_per_p'].transform('std')

# Processing data_ge
data_ge = data.copy()

for i in range(2):
    trial_filter = range(15, 189) if i == 0 else range(25, 181)

    data_ge[i] = data_ge[i][data_ge[i]['trials'].isin(trial_filter)]
    data_ge[i]['mean_geus_category'] = data_ge[i].groupby(['stimulus', 'category'])['US_expect'].transform('mean')

    data_ge[i]['mean_geus'] = data_ge[i].groupby(['stimulus', 'participant'])['US_expect'].transform('mean')
    data_ge[i]['mean_per'] = data_ge[i].groupby(['stimulus', 'participant'])['Per_size'].transform('mean')

    data_ge[i]['mean_geus_all'] = data_ge[i].groupby(['stimulus'])['US_expect'].transform('mean')
    data_ge[i]['sd_geus_all'] = data_ge[i].groupby(['stimulus'])['US_expect'].transform('std')

# Generalization: All data - generalization pattern
# geplot_all contains plots for each experiment
def create_geplot(data_ge, stimulus_levels, titles):
    plots = []
    for a in range(2):
        plot_data = data_ge[a]
        plt.figure(figsize=(6, 4))
        
        # Plot mean_geus_all
        plt.plot(
            plot_data['stimulus'], 
            plot_data['mean_geus_all'], 
            linewidth=1, 
            label='Mean Geus All', 
            color='blue'
        )
        
        # Plot mean_geus per participant
        for participant in plot_data['participant'].unique():
            participant_data = plot_data[plot_data['participant'] == participant]
            plt.plot(
                participant_data['stimulus'], 
                participant_data['mean_geus'], 
                linewidth=0.5, 
                alpha=0.5, 
                color='grey'
            )
        
        # Add scatter points
        plt.scatter(
            plot_data['stimulus'], 
            plot_data['mean_geus_all'], 
            s=20, 
            label='Mean Geus All Points', 
            color='red'
        )
        plt.scatter(
            plot_data['stimulus'], 
            plot_data['mean_geus'], 
            s=10, 
            alpha=0.5, 
            color='grey'
        )

        # Customize the plot
        plt.title(titles[a])
        plt.xlabel('Stimulus')
        plt.ylabel('US Expectancy')
        plt.ylim(1, 10)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plots.append(plt)
    return plots

titles = [
    "Experiment 1 (simple conditioning)", 
    "Experiment 2 (differential conditioning)"
]

generalization_plots = create_geplot(data_ge, [stimulus_level_study1, stimulus_level_study2], titles)

for idx, plot in enumerate(generalization_plots):
    plot.savefig(f"{plot_folder}Ge_Ex{idx + 1}.png", dpi=300, width=4, height=4)

# plt.show()


# Group - Generalization Pattern
def create_phyge_group_plot(data_ge, group, titles):
    plots = []
    for a in range(2):
        plot_data = data_ge[a]
        
        plt.figure(figsize=(6, 4))
        
        # Facet by group
        for category in group:
            category_data = plot_data[plot_data['category'] == category]
            
            plt.plot(
                category_data['stimulus'], 
                category_data['mean_geus_category'], 
                linewidth=1, 
                label=f'{category} - Mean Geus Category', 
                color='blue'
            )
            plt.plot(
                category_data['stimulus'], 
                category_data['mean_geus'], 
                linewidth=0.5, 
                alpha=0.5, 
                color='grey'
            )
            plt.scatter(
                category_data['stimulus'], 
                category_data['mean_geus'], 
                s=10, 
                alpha=0.5, 
                color='grey'
            )
            plt.scatter(
                category_data['stimulus'], 
                category_data['mean_geus_category'], 
                s=20, 
                color='red'
            )

        plt.title(titles[a])
        plt.xlabel('Stimulus')
        plt.ylabel('US Expectancy')
        plt.ylim(1, 10)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plots.append(plt)
    return plots

group_titles = [
    "Generalized responding: Experiment 1", 
    "Generalized responding: Experiment 2"
]

group = ["m = 1", "m = 2", "m = 3", "m = 4"]  # Replace with actual group categories
phyge_group_plots = create_phyge_group_plot(data_ge, group, group_titles)

# Optionally save plots
for idx, plot in enumerate(phyge_group_plots):
    plot.savefig(f"{plot_folder}Group_Ex{idx + 1}.png", dpi=300, width=4, height=4)

# plt.show()

# All data - Learning Pattern
def create_lr_group_plot(data, group, titles):
    plots = []
    for a in range(2):
        plot_data = data[a]
        plt.figure(figsize=(8, 6))
        
        for category in group:
            category_data = plot_data[(plot_data['category'] == category) & 
                                      (plot_data['stimulus'].isin(["CS+", "CS-"])) & 
                                      (plot_data['trials'].isin(range(1, 189) if a == 0 else range(1, 181)))]

            plt.plot(
                category_data['trials'], 
                category_data['mean_us_trial'], 
                label=f'{category} - Mean US Trial', 
                linewidth=1
            )
            plt.scatter(
                category_data['trials'], 
                category_data['mean_us_trial'], 
                s=15
            )
            plt.plot(
                category_data['trials'], 
                category_data['US_expect'], 
                alpha=0.1, 
                linewidth=0.5
            )
            plt.scatter(
                category_data['trials'], 
                category_data['US_expect'], 
                alpha=0.1, 
                s=10
            )
        
        plt.title(titles[a])
        plt.xlabel('Trials')
        plt.ylabel('US Expectancy')
        plt.ylim(1, 10)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plots.append(plt)
    return plots

learning_titles = [
    "Learning: Experiment 1", 
    "Learning: Experiment 2"
]

lr_group_plots = create_lr_group_plot(data, group, learning_titles)

# Optionally save plots
for idx, plot in enumerate(lr_group_plots):
    plot.savefig(f"{plot_folder}Learning_Ex{idx + 1}.png", dpi=300, width=4, height=4)
# plt.show()


# Reading data from .pkl files
alpha = [
    [
        0.0002413973, 0.1503184450 ,0.0000000000 ,0.1153081286, 0.0000000000,
        0.0000000000 ,0.0616393906, 0.0003600646 ,0.2827100645, 0.0000000000,
        0.0000000000, 0.1265856153, 0.0737969477, 0.1169614775, 0.1429669124,
        0.0436385047, 0.1287666352, 0.2432256665, 0.0000000000, 0.0000000000,
        0.0000000000, 0.0000000000, 0.0489073623, 0.0000000000, 0.0002392894,
        0.2653869115, 0.0000000000, 0.1973438437, 0.0557421839, 0.0000000000,
        0.0000000000 ,0.1917315484 ,0.0023534087, 0.0001420344, 0.1461624912,
        0.0112136777, 0.0032220487 ,0.4682102110, 0.0000000000, 0.1271594485,
    ],
    [
        0.18135081, 0.10873111, 0.00000000, 0.47621854, 0.41804523, 0.01574880,
        0.00000000,0.37195948, 0.23287736, 0.31847941 ,0.06411720 ,0.11634914,
        0.08054728, 0.01896177 ,0.00000000, 0.51414218, 0.52899111, 0.48451255,
        0.26109789 ,0.23190834 ,0.13631174, 0.39675650, 0.36286111, 0.06817227,
        0.12631234, 0.58035596 ,0.00353801 ,0.24735968 ,0.11325343, 0.10944526,
        0.31704690, 0.46640846 ,0.29432879 ,0.11076664 ,0.21861952 ,0.24516812,
        0.00000000, 0.04874823, 0.60446267 ,0.06527241   
    ]
]

jags1 = pd.read_pickle(f'{root_folder}Data/Data1_JAGSinput_CLG.pkl')
jags2 = pd.read_pickle(f'{root_folder}Data/Data2_JAGSinput_CLG.pkl')

v1 = np.zeros((jags1['r'].shape[0], jags1['r'].shape[1] + 1))
v2p = np.zeros((jags2['r_plus'].shape[0], jags2['r_plus'].shape[1] + 1))
v2m = np.zeros((jags2['r_plus'].shape[0], jags2['r_plus'].shape[1] + 1))

for i in range(jags1['r'].shape[0]):  # Participant loop
    for j in range(jags1['r'].shape[1]):  # Trial loop
        v1[i, j + 1] = (
            v1[i, j] + alpha[0][i] * (jags1['r'].iloc[i, j] - v1[i, j])
            if jags1['k'].iloc[i, j] == 1
            else v1[i, j]
        )

for i in range(jags2['r_plus'].shape[0]):  # Participant loop
    for j in range(jags2['r_plus'].shape[1]):  # Trial loop
        v2p[i, j + 1] = (
            v2p[i, j] + alpha[1][i] * (jags2['r_plus'].iloc[i, j] - v2p[i, j])
            if jags2['k_plus'].iloc[i, j] == 1
            else v2p[i, j]
        )
        v2m[i, j + 1] = (
            v2m[i, j] + alpha[1][i] * (jags2['r_minus'].iloc[i, j] - v2m[i, j])
            if jags2['k_minus'].iloc[i, j] == 1
            else v2m[i, j]
        )

vp1 = pd.melt(pd.DataFrame(v1), var_name='trials', value_name='vp')
vp1['participant'] = vp1['trials'] // v1.shape[1]

vp2 = pd.melt(pd.DataFrame(v2p), var_name='trials', value_name='vp')
vp2['participant'] = vp2['trials'] // v2p.shape[1]

vm2 = pd.melt(pd.DataFrame(v2m), var_name='trials', value_name='vm')
vm2['participant'] = vm2['trials'] // v2m.shape[1]

# Merging v1, v2p, and v2m into data
for df, col in zip([vp1, vp2, vm2], ['vp', 'vp', 'vm']):
    data[0 if col == 'vp' else 1] = data[0 if col == 'vp' else 1].merge(df, on=['participant', 'trials'], how='left')

data[0]['mean_vp'] = data[0].groupby(['category', 'trials'])['vp'].transform('mean')
data[1]['mean_vp'] = data[1].groupby(['category', 'trials'])['vp'].transform('mean')
data[1]['mean_vm'] = data[1].groupby(['category', 'trials'])['vm'].transform('mean')

# Line colors for plotting
linecolors = {'vp': "#e41a1c", 'vm': "#377eb8"}

# Learning Plot for Experiment 1
plt.figure(figsize=(8, 6))
for category in group:
    category_data = data[0][data[0]['category'] == category]
    plt.plot(
        category_data['trials'], 
        category_data['mean_vp'], 
        label=f'{category} - Mean VP', 
        color=linecolors['vp'], 
        linewidth=2
    )
plt.title('Learning: Experiment 1')
plt.xlabel('Trials')
plt.ylabel('Associative Strengths')
plt.legend()
plt.grid(True)
plt.show()

# Learning Plot for Experiment 2
plt.figure(figsize=(8, 6))
for category in group:
    category_data = data[1][data[1]['category'] == category]
    plt.plot(
        category_data['trials'], 
        category_data['mean_vp'], 
        label=f'{category} - Mean VP', 
        color=linecolors['vp'], 
        linewidth=2
    )
    plt.plot(
        category_data['trials'], 
        category_data['mean_vm'], 
        label=f'{category} - Mean VM', 
        color=linecolors['vm'], 
        linewidth=2
    )
plt.title('Learning: Experiment 2')
plt.xlabel('Trials')
plt.ylabel('Associative Strengths')
plt.legend()
plt.grid(True)
plt.show()




# Lambda posterior



lambdavals = [
    pd.DataFrame(result[0]['sims_list']['lambda']).melt(value_name="value"),
    pd.DataFrame(result[1]['sims_list']['lambda']).melt(value_name="value"),
]
lambdaposterior = {
    "study1": lambdavals[0]["value"].values,
    "study2": lambdavals[1]["value"].values,
}

# Plot 1
plot1_data = lambdaposterior['study1'][lambdaposterior['study1'] <= 0.3]
plt.figure(figsize=(6, 4))
sns.histplot(plot1_data['study1'], bins=6, color='red', alpha=0.5, kde=False)
plt.axvline(x=0.0052, color='black', linestyle='--', linewidth=1.5)
plt.xticks([0.0052, 0.01, 0.1, 0.2, 0.3])
plt.xlabel('λ posterior')
plt.ylabel('Posterior samples')
plt.title('Experiment 1')
plt.grid(True)
plt.show()

# Plot 2
plot2_data = lambdaposterior['study2'][lambdaposterior['study2'] <= 0.3]
plt.figure(figsize=(6, 4))
sns.histplot(plot2_data['study2'], bins=6, color='red', alpha=0.5, kde=False)
plt.axvline(x=0.0052, color='black', linestyle='--', linewidth=1.5)
plt.xticks([0.0052, 0.01, 0.1, 0.2, 0.3])
plt.xlabel('λ posterior')
plt.ylabel('Posterior samples')
plt.title('Experiment 2')
plt.grid(True)
plt.show()

# Plot 3
plot3_data = lambdaposterior['study1'][lambdaposterior['study1'] <= 0.01]
plt.figure(figsize=(6, 4))
sns.histplot(plot3_data['study1'], bins=6, color='red', alpha=0.5, kde=False)
plt.axvline(x=0.0052, color='black', linestyle='--', linewidth=1.5)
plt.xticks([0, 0.0052, 0.01])
plt.xlabel('λ posterior')
plt.ylabel('Posterior samples')
plt.grid(True)
plt.show()

# Plot 4
plot4_data = lambdaposterior['study2'][lambdaposterior['study2'] <= 0.01]
plt.figure(figsize=(6, 4))
sns.histplot(plot4_data['study2'], bins=6, color='red', alpha=0.5, kde=False)
plt.axvline(x=0.0052, color='black', linestyle='--', linewidth=1.5)
plt.xticks([0, 0.0052, 0.01])
plt.xlabel('λ posterior')
plt.ylabel('Posterior samples')
plt.grid(True)
plt.show()


# All data - perception pattern
data[0]['d_phy_p'] = data[0]['dphy']
data[0]['d_per_p'] = data[0]['dper']

# Perception Mean Group Plot
permean_group_plot = []
for a in range(2):
    experiment_plots = []
    for b in range(4):
        subset = data[a][data[a]['category'] == group[b]]
        plt.figure(figsize=(6, 4))
        plt.plot(subset['stimulus'], subset['mean_disp_group'], label='Mean Disp Group', linewidth=1)
        plt.scatter(subset['stimulus'], subset['mean_disp_group'], s=30, label='Mean Disp Group Points')
        plt.plot(subset['stimulus'], subset['mean_disp_indi'], alpha=0.1, label='Mean Disp Individual', linewidth=0.5)
        plt.scatter(subset['stimulus'], subset['mean_disp_indi'], s=10, alpha=0.1, label='Mean Disp Individual Points')
        plt.ylim(0, max(data[0]['mean_disp_indi'].max(), data[1]['mean_disp_indi'].max()))
        plt.xticks(rotation=45)
        plt.title(f"Perception: {group[b]}")
        plt.xlabel("Stimulus")
        plt.ylabel("Distance to CS+")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        experiment_plots.append(plt)
    permean_group_plot.append(experiment_plots)

# Perception SD Group Plot
persd_group_plot = []
for a in range(2):
    experiment_plots = []
    for b in range(4):
        subset = data[a][data[a]['category'] == group[b]]
        plt.figure(figsize=(6, 4))
        plt.plot(subset['stimulus'], subset['sd_disp_group'], label='SD Disp Group', linewidth=1)
        plt.scatter(subset['stimulus'], subset['sd_disp_group'], s=30, label='SD Disp Group Points')
        plt.plot(subset['stimulus'], subset['sd_disp_indi'], alpha=0.1, label='SD Disp Individual', linewidth=0.5)
        plt.scatter(subset['stimulus'], subset['sd_disp_indi'], s=10, alpha=0.1, label='SD Disp Individual Points')
        plt.ylim(0, max(data[0]['sd_disp_indi'].max(), data[1]['sd_disp_indi'].max()))
        plt.xticks(rotation=45)
        plt.title(f"Perception SD: {group[b]}")
        plt.xlabel("Stimulus")
        plt.ylabel("Distance to CS+ (SD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        experiment_plots.append(plt)
    persd_group_plot.append(experiment_plots)

# Combine Plots for Perception
per_plot = []
for a in range(2):
    experiment_plots = []
    for b in range(4):
        # Here you would combine permean_group_plot and persd_group_plot visually, e.g., using subplots
        fig, ax = plt.subplots(2, 1, figsize=(7, 10))
        # Mean plot
        ax[0].plot(permean_group_plot[a][b].gca().lines[0].get_data()[0], 
                   permean_group_plot[a][b].gca().lines[0].get_data()[1], 
                   label='Mean Disp Group', linewidth=1)
        # SD plot
        ax[1].plot(persd_group_plot[a][b].gca().lines[0].get_data()[0], 
                   persd_group_plot[a][b].gca().lines[0].get_data()[1], 
                   label='SD Disp Group', linewidth=1)
        
        ax[0].set_title(f"Perception Mean: {group[b]}")
        ax[1].set_title(f"Perception SD: {group[b]}")
        plt.tight_layout()
        experiment_plots.append(fig)
    per_plot.append(experiment_plots)

# Optionally save plots
for a in range(2):
    for b in range(4):
        per_plot[a][b].savefig(f"{plot_folder}Per_Ex{a+1}_{group[b]}.png", dpi=300, width=7, height=7)

# Similarity Simulation
s1_lambda_mean = [
    0.140421599 ,0.209843811, 0.000000000, 0.116782492 ,0.000000000 ,0.000000000,
    0.115307936 ,0.114540574 ,0.047684934 ,0.000000000 ,0.000000000, 0.019768497,
    0.100081565 ,0.094810481 ,0.103743340 ,0.004543662 ,0.114755174, 0.161412134,
    0.000000000, 0.000000000 ,0.014858194, 0.000000000 ,0.044342975, 0.000000000,
    0.112632434, 0.116114159, 0.000000000, 0.109718856, 0.167917467 ,0.000000000,
    0.000000000 ,0.069590840 ,0.132579438 ,0.131603824, 0.098998868 ,0.004949202,
    0.305727750 ,0.065799351, 0.000000000 ,0.171632302
]
s2_lambda_q50 = [
    0.150092822, 0.010169454 ,0.000000000 ,0.052278416, 0.044775165, 0.126712645,
    0.000000000, 0.012312330 ,0.066893686 ,0.246951466, 0.193344527 ,0.024535229,
    0.001875938, 0.010112683 ,0.000000000 ,0.056616452, 0.275101698 ,0.037795311,
    0.095709303, 0.160095126 ,0.065070314 ,0.060611868, 0.290676368 ,0.025419314,
    0.005168648, 0.209129830 ,0.004526049 ,0.052049602, 0.031310934 ,0.023300280,
    0.133731134, 0.077120380 ,0.113047135, 0.035448221, 0.073786051 ,0.082171288,
    0.000000000, 0.001998707 ,0.247051407 ,0.002566278

]

mean_lam = {
    "s1": pd.DataFrame({
        "participant": [str(i) for i in range(1, 41)],
        "lambda": s1_lambda_mean
    }),
    "s2": pd.DataFrame({
        "participant": [str(i) for i in range(1, 41)],
        "lambda":s2_lambda_q50
    })
}

# Adding lambda values to the data
for i in range(2):
    data[i] = data[i].merge(mean_lam[f"s{i + 1}"], on="participant", how="left")

# Similarity Simulation Plot
sim_plot = []
for x in range(2):
    data[x]['sim_similarity'] = np.where(
        data[x]['category'] == "Non-Learners",
        1,
        np.where(
            data[x]['category'] == "Perceptual Generalizers",
            np.exp(-data[x]['lambda'] * data[x]['d_per_p']),
            np.exp(-data[x]['lambda'] * data[x]['d_phy_p'])
        )
    )
    data[x]['stim'] = np.where(
        data[x]['stimulus'] == "CS+",
        "CS+",
        np.where(data[x]['stimulus'] == "CS-", "CS-", "TS")
    )

    plt.figure(figsize=(10, 8))
    categories = data[x]['category'].unique()
    for category in categories:
        subset = data[x][data[x]['category'] == category]
        plt.plot(
            subset['d_phy_p'] + 10, subset['sim_similarity'],
            color='grey', alpha=0.25, label=f'{category} - Physical'
        )
        plt.scatter(
            subset['d_phy_p'] + 10, subset['sim_similarity'],
            color='red', s=10
        )
        plt.plot(
            -subset['d_per_p'] - 10, subset['sim_similarity'],
            color='grey', alpha=0.25, label=f'{category} - Perceptual'
        )
        plt.scatter(
            -subset['d_per_p'] - 10, subset['sim_similarity'],
            color='blue', s=10
        )

    plt.axvline(x=0, color='black', linestyle='dashed', linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Distance to CS+ (left: perceptual; right: physical)")
    plt.ylabel("Similarity to CS+")
    plt.title("Similarity")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.tight_layout()
    sim_plot.append(plt)

# Optionally save the plots
for i, plot in enumerate(sim_plot):
    plot.savefig(f"${plot_folder}Similarity_Ex{i + 1}.png", dpi=300)


# plt.show()


# Function to combine and save plots for each experiment
for i in range(2):
    # Create a new PDF file to save the combined plots
    pdf_filename = f"Plots/ResPattern_{i+1}.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Combine the plots
        fig, axes = plt.subplots(3, 2, figsize=(17, 15), gridspec_kw={'height_ratios': [0.3, 0.35, 0.35]})
        
        # First row: geplot_all and phyge_group_plot
        axes[0, 0].imshow(geplot_all[i].canvas.buffer_rgba())
        axes[0, 1].imshow(phyge_group_plots[i].canvas.buffer_rgba())
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

        # Second row: lr_group_plot
        axes[1, 0].imshow(lr_group_plots[i].canvas.buffer_rgba())
        axes[1, 1].axis('off')

        # Third row: sim_plot
        axes[2, 0].imshow(sim_plot[i].canvas.buffer_rgba())
        axes[2, 1].axis('off')

        # Add annotations
        fig.suptitle("Results Pattern", fontsize=20, fontweight="bold")
        plt.subplots_adjust(hspace=0.3)

        # Save to PDF
        # pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved combined plot to {pdf_filename}")
