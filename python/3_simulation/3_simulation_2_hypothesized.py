
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle



root_folder = "../../"
plts_folder = root_folder + "Plots/py/3_HypothesizedParticipantsPlots/"
vars_folder = root_folder + "vars/py/"



# ## Part 2: 200 hypothesized participants
# ### Load data
# 


def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
data = load_data_from_pickle("../../Data/res_py/Data_s2.pkl")
data = data[data['participant'] == 1]

data = data.sort_values(by=['participant', 'trials'])



# ### Lambda value
# Here we show how we decide the boundary of λ value


max_d = data['d_phy_p'].max()
min_d = sorted(data['d_phy_p'].unique())[1]

# Create decay values table for different lambda values
lambda_values = np.arange(0, 0.501, 0.001)
sim_decay = pd.DataFrame({
    'lambda': lambda_values,
    'sim_max': np.exp(-lambda_values * max_d),
    'sim_min': np.exp(-lambda_values * min_d)
})

# Define similarity thresholds
min_perc = 0.7
max_perc = 0.95

# Calculate the minimum and maximum lambda values
min_lam = round(-(np.log(min_perc) / max_d), 4)
max_lam = round(-(np.log(max_perc) / min_d), 4)

# Create decay table for different distances with min and max lambda values
d_values = np.arange(0, 101, 1)
sim_decay_2 = pd.DataFrame({
    'd': d_values,
    'sim_max': np.exp(-min_lam * d_values),
    'sim_min': np.exp(-max_lam * d_values),
    'sim_min_og': np.exp(0 * d_values)
})

# Plot decay of similarity for different lambda values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for sim_decay lambda values
sns.lineplot(data=sim_decay, x='lambda', y='sim_max', ax=axes[0], linewidth=2)
axes[0].axvline(min_lam, linestyle='--', color='grey')
axes[0].axhline(min_perc, linestyle='--', color='grey')
axes[0].plot(min_lam, min_perc, 'ro', markersize=8)
axes[0].set_xlabel("λ")
axes[0].set_ylabel(f"Similarity given distance = {max_d}")
axes[0].text(min_lam + 0.1, min_perc + 0.05, f"λ = {min_lam}", color='red')

# Plot for sim_decay_2 values by distance
sns.lineplot(data=sim_decay_2, x='d', y='sim_max', ax=axes[1], color='black', linewidth=2)
axes[1].text(40, 0.95, f"λ = {min_lam}", color='black')
axes[1].set_ylim(0, 1)
axes[1].set_xticks(np.arange(0, 101, 25))
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Similarity")

# Combine and save plots
plt.suptitle("Decay of Similarity with Different Lambda Values")
plt.savefig(f"{plts_folder}/Sim_lambda.pdf", dpi=300, format="pdf")
plt.show()


# ### 1. Parameter settings
# We will simulate hypothesised participants in 4 latent groups.
# 
# 


# Define the number of participants in each group
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




# ### 2. Extract variables
# 


def extract_fun(var_key, var_val):
    _x = {}
    for x in range(Ntrials):
        col_name = f"{var_key}.{x + 1}"
        _col_data = []
        for nc in range(Nparticipants):
            _col_data.append(data[var_val].iloc[x])
        # print(_col_data)
        _x[col_name]= _col_data
    return pd.DataFrame(_x) 




# create a new list 'variable_name' with named elements that correspond to the names of variables in the 'data' list
variable_name = {
  "d_phy_p" : "d_phy_p", 
  "d_phy_m" : "d_phy_m", 
  "kplus" :"CSptrials", 
  "kminus" : "CSmtrials", 
  "rplus" :"USp", 
  "rminus" :"USm"
}
# Apply 'extract_fun' to each element in 'variable_name' to create the 'variables' dictionary
variables = pd.concat([extract_fun(var, variable_name[var]) for var in variable_name], axis=1)

variables.to_csv(f"{vars_folder}variables_py.csv")


# This is a function to get the values of columns and make them in one column dataframe that I can use after, the count of each one should be 36,000 (180 * 200):


def getValuesOfCols(col_start):
    _cols = [col for col in variables.columns if col.startswith(col_start)]
    vals = pd.concat([variables[col] for col in _cols], ignore_index=True).to_frame(name=col_start)
    return vals


d_phy_p_vals = getValuesOfCols("d_phy_p")
d_phy_m_vals = getValuesOfCols("d_phy_m")
rplus_vals = getValuesOfCols("rplus")

vars = {
    "rplus": np.empty((Nparticipants, Ntrials)),
    "kplus": np.empty((Nparticipants, Ntrials)),
    "kminus": np.empty((Nparticipants, Ntrials)),
    "rminus": np.empty((Nparticipants, Ntrials)),
    "d_phy_p": np.empty((Nparticipants, Ntrials)),
    "d_phy_m": np.empty((Nparticipants, Ntrials)),
    
}

for i in range(Nparticipants):
    for j in range(Ntrials):
        vars["rplus"][i,j] = variables[f'rplus.{j + 1}'][i]
        vars["kplus"][i,j] = variables[f'kplus.{j + 1}'][i]
        vars["kminus"][i,j] = variables[f'kminus.{j + 1}'][i]
        vars["rminus"][i,j] = variables[f'rminus.{j + 1}'][i]
        vars["d_phy_p"][i,j] = variables[f'd_phy_p.{j + 1}'][i]
        vars["d_phy_m"][i,j] = variables[f'd_phy_m.{j + 1}'][i]






# ### 3. Simulation
# 


# These lines of code define a python function called lg_fun() that appears to simulate a learning process. 
# The function has three parameters: persd, A, and K. 
# The persd parameter represents the perceived distance, and the A and K parameters appear to be constants.
# 
# Inside the function, several matrices and vectors are initialized. 
# The y matrix will hold the output of the learning process. 
# The theta matrix will hold some intermediate values used in the learning process. 
# The s_plus and s_minus matrices will hold similarity values. 
# The v_plus and v_minus matrices will hold some other intermediate values used in the learning process. 
# The g matrix will hold generalization values. 
# The d_per_p and d_per_m matrices will hold perceived distances. 
# The group vector will hold group membership information for each participant. 
# The alpha, lambda, w0, w1, and sigma vectors will hold parameters that vary across participants.
# 
# The function then loops over all participants and trials, and performs the following actions:
# 1.Updates the v_plus and v_minus matrices based on learning. 
# 2. Calculates the s_plus and s_minus matrices based on similarity. 
# 3. Calculates the g matrix based on generalization. 
# 4. Calculates the theta matrix based on the g matrix and the A and K parameters. 
# 5. Calculates the y matrix based on the theta matrix and some other intermediate values. 
# 6. Finally, the function returns the y matrix as its output.


v_plus = np.zeros((Nparticipants, Ntrials + 1))
alpha = pd.read_csv('../../vars/randoms/alpha.csv').values.tolist()
alpha = [it[0] for it in  alpha]
group = np.repeat([1, 2, 3, 4], Nparticipants // 4)
v_minus = np.zeros((Nparticipants, Ntrials + 1))
for i in range(Nparticipants):
    for j in range(Ntrials):
        # Learning
        v_plus[i, j + 1] = (
            v_plus[i, j] + alpha[i] * ( vars['rplus'][i, j] - v_plus[i, j])
            if group[i] != 1 and vars['kplus'][i, j] == 1
            else v_plus[i, j]
            if group[i] != 1
            else 0
        )
        v_minus[i, j + 1] = (
            v_minus[i, j] + alpha[i] * (vars['rminus'][i, j] - v_minus[i, j])
            if group[i] != 1 and vars['kminus'][i, j] == 1
            else v_minus[i, j]
            if group[i] != 1
            else 0
        )

v_plus = pd.DataFrame(v_plus)
v_plus = pd.melt(v_plus.iloc[:, :180])['value']



def lg_fun(persd=30, A=1, K=10):
    # Initialize matrices and arrays
    y = np.zeros((Nparticipants, Ntrials))
    theta = np.zeros((Nparticipants, Ntrials))
    s_plus = np.zeros((Nparticipants, Ntrials))
    s_minus = np.zeros((Nparticipants, Ntrials))
    v_plus = np.zeros((Nparticipants, Ntrials + 1))
    v_minus = np.zeros((Nparticipants, Ntrials + 1))
    g = np.zeros((Nparticipants, Ntrials))
    d_per_p = np.zeros((Nparticipants, Ntrials))
    d_per_m = np.zeros((Nparticipants, Ntrials))
    
    
    group = np.repeat([1, 2, 3, 4], Nparticipants // 4)
    
    # Initialize participant-specific parameters
    alpha = np.zeros(Nparticipants)
    lambd = np.zeros(Nparticipants)
    w0 = np.zeros(Nparticipants)
    w1 = np.zeros(Nparticipants)
    sigma = np.zeros(Nparticipants)
    
    
 
    for i in range(Nparticipants):
        alpha[i] = 0 if group[i] == 1 else np.random.beta(1, 1)
        #set lambda[i]
        if group[i] == 1:
            lambd[i] = 0
        elif group[i] == 2:
            lambd[i] = max(0, min(0.0052, np.random.normal(0.0026, 0.001)))
        else:
            lambd[i] = max(0.0052, min(0.3022, np.random.normal((0.0052 + 0.3022) / 2, 0.1)))
        w0[i] = np.random.normal(0, 5) if group[i] < 3 else np.random.normal(-2, 1)
        w1[i] = np.random.gamma(10, 1)
        sigma[i] = 2.5 if group[i] == 1 else 0.5
        
    for i in range(Nparticipants):
        for j in range(Ntrials):
            # Learning
            v_plus[i, j + 1] = (
                v_plus[i, j] + alpha[i] * (vars['rplus'][i, j] - v_plus[i, j])
                if group[i] != 1 and vars['kplus'][i, j] == 1
                else v_plus[i, j]
            )
            v_minus[i, j + 1] = (
                v_minus[i, j] + alpha[i] * (vars['rminus'][i, j] - v_minus[i, j])
                if group[i] != 1 and vars['kminus'][i, j] == 1
                else v_minus[i, j]
            )

            # Similarity
            d_per_p[i, j] = max(0, np.random.normal(vars['d_phy_p'][i, j], persd))
            d_per_m[i, j] = max(0, np.random.normal(vars['d_phy_m'][i, j], persd))

            s_plus[i, j] = (
                np.exp(-lambd[i] * d_per_p[i, j])
                if v_plus[i, j] > 0 and group[i] > 1 and group[i] == 4
                else np.exp(-lambd[i] * vars['d_phy_p'][i, j])
                if v_plus[i, j] > 0 and group[i] > 1
                else 1
            )
            s_minus[i, j] = (
                np.exp(-lambd[i] * d_per_m[i, j])
                if abs(v_minus[i, j]) > 0 and group[i] > 1 and group[i] == 4
                else np.exp(-lambd[i] * vars['d_phy_m'][i, j])
                if abs(v_minus[i, j]) > 0 and group[i] > 1
                else 1
            )

            # Generalization
            g[i, j] = v_plus[i, j] * s_plus[i, j] + v_minus[i, j] * s_minus[i, j]
            theta[i, j] = A + (K - A) / (1 + np.exp(-(w0[i] + w1[i] * g[i, j])))

            # Response
            y[i, j] = np.random.normal(theta[i, j], sigma[i])
            
       
    rplus = [it[0] for it in  rplus_vals.values]
    d_phy_p = [it[0] for it in  d_phy_p_vals.values]
    d_phy_m = [it[0] for it in  d_phy_m_vals.values]
    
    
    
    data = pd.DataFrame({
        "participant": np.repeat(np.arange(1, Nparticipants + 1), Ntrials),
        "trials": np.tile(np.arange(1, Ntrials + 1), Nparticipants),
        "group": np.repeat(group, Ntrials),
        "alpha": np.repeat(alpha, Ntrials),
        "lambda": np.repeat(lambd, Ntrials),
        "w0": np.repeat(w0, Ntrials),
        "sigma": np.repeat(sigma, Ntrials),
        "w1": np.repeat(w1, Ntrials),
        "y": y.flatten(),
        "g": g.flatten(),
        "theta": theta.flatten(),
        "v_plus": v_plus[:, :Ntrials].flatten(),
        "v_minus": v_minus[:, :Ntrials].flatten(),
        "s_plus": s_plus.flatten(),
        "s_minus": s_minus.flatten(),
        "d_per_p": d_per_p.flatten(),
        "d_per_m": d_per_m.flatten(),
        "d_phy_p": d_phy_p,
        "d_phy_m": d_phy_m,
        "rplus": rplus
    })

    # Add stimulus labels
    data["stim"] = np.where(
        data["d_phy_p"] == 0, "CS+",
        np.where(data["d_phy_m"] == 0, "CS-", "TS")
    )

    return data        


result = lg_fun()


group_labels = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers"]
result['group'] = pd.Categorical(result['group'], categories=[1, 2, 3, 4], ordered=True).rename_categories(group_labels)
result = result2.merge(data[['trials', 'stimulus']], on='trials', how='left')
result['stimulus'] = pd.Categorical(result['stimulus'], categories=["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"], ordered=True)




# ## 4. Visulization
# 


# ### 4.2 Leanring


# parsample = np.concatenate([
#     np.random.choice(range(1, 51), 5, replace=False),
#     np.random.choice(range(51, 101), 5, replace=False),
#     np.random.choice(range(101, 151), 5, replace=False),
#     np.random.choice(range(151, 201), 5, replace=False)
# ])
parsample = np.array(pd.read_csv("../../vars/randoms/parsample.csv").values)
parsample = parsample.flatten()
linecolors = {'vp': "#e41a1c", 'vm': "#377eb8"}



def create_learning_plt():
    # Filter and group the data
    L = (result[result['trials'] < 24]
        .groupby(['trials', 'group'])
        .agg(mean_vp=('v_plus', 'mean'), mean_vm=('v_minus', 'mean'))
        .reset_index())

    # Set up the facet grid
    unique_groups = result['group'].unique()
    num_groups = len(unique_groups)
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=True, sharey=True)

    for i, (group, ax) in enumerate(zip(unique_groups, axes)):
        # Filter data for the current group
        group_data = result[(result['group'] == group) & (result['trials'] < 24)]
        group_means = L[L['group'] == group]

        # Plot individual lines and points for v_plus
        sns.lineplot(data=group_data, x='trials', y='v_plus', hue='participant',
                    palette=["#e41a1c"], ax=ax, alpha=0.1, legend=None)
        sns.scatterplot(data=group_data, x='trials', y='v_plus', hue='participant',
                        palette=["#e41a1c"], ax=ax, alpha=0.1, s=10, legend=None)

        # Plot individual lines and points for v_minus
        sns.lineplot(data=group_data, x='trials', y='v_minus', hue='participant',
                    palette=["#377eb8"], ax=ax, alpha=0.3, legend=None)
        sns.scatterplot(data=group_data, x='trials', y='v_minus', hue='participant',
                        palette=["#377eb8"], ax=ax, alpha=0.3, s=10, legend=None)

        # Plot mean lines and points for v_plus (mean_vp) and v_minus (mean_vm)
        sns.lineplot(data=group_means, x='trials', y='mean_vp', color='red', linewidth=3, label='CS+ strength', ax=ax)
        sns.scatterplot(data=group_means, x='trials', y='mean_vp', color='red', s=50, legend=None, ax=ax)
        sns.lineplot(data=group_means, x='trials', y='mean_vm', color='blue', linewidth=3, label='CS- strength', ax=ax)
        sns.scatterplot(data=group_means, x='trials', y='mean_vm', color='blue', s=50, legend=None, ax=ax)

        # Shape and color for 'rplus' and 'stim'
        for _, row in group_data.iterrows():
            # Bottom layer for 'stim' - adjusting y position to -1.4 as in R code
            ax.scatter(row['trials'], -1.4, color='red' if row['stim'] == 'CS+' else 'blue', 
                    s=25, marker='s', edgecolor='black')

        # Customize labels and title
        ax.set_title(group, loc='center')
        ax.set_ylabel("Associative strengths")

    # Shared x-axis label
    axes[-1].set_xlabel("Trials")

    # Set y-axis limits and breaks
    plt.setp(axes, ylim=(-1.5, 1), yticks=[-1, -0.5, 0, 0.5, 1])

    # Custom legend for shape encoding
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Shock', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Shock absent', markerfacecolor='black', markersize=8),
        Line2D([0], [0], color='red', lw=3, label='CS+ strength'),
        Line2D([0], [0], color='blue', lw=3, label='CS- strength'),
        Line2D([0], [0], marker='s', color='red', label='CS+ trial', markersize=8, markerfacecolor='red', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='blue', label='CS- trial', markersize=8, markerfacecolor='blue', markeredgecolor='black')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # Apply theme and styling
    sns.set(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False})

    # Overall title
    plt.suptitle("Learning", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{plts_folder}Hypothesized_Learning.png", dpi=300, bbox_inches="tight")

    plt.show()

    return plt 




plt1 = create_learning_plt()



# ### 4.3 Similarity
# 


def do_sim_similarity_plt():
    # Calculate max value for `d_per_p` and add 10
    max_d = result['d_per_p'].max() + 10

    # Filter rows where group is "Non-Learners" or (v_plus > 0.1 and v_minus < 0.1)
    filtered_result = result[(result['group'] == "Non-Learners") | ((result['v_plus'] > 0.1) & (result['v_minus'] < 0.1))]

    # Set up the plot
    sns.set(style="whitegrid")
    g = sns.FacetGrid(filtered_result, col="group", col_wrap=1, height=4, sharey=True)

    # Plot each group
    for ax, (group, group_data) in zip(g.axes.flat, filtered_result.groupby("group")):
        # Plot grey lines for s_plus as a function of `d_phy_p + 10` and `-d_per_p - 10`
        ax.plot(group_data['d_phy_p'] + 10, group_data['s_plus'], color="grey", alpha=0.25)
        ax.plot(-group_data['d_per_p'] - 10, group_data['s_plus'], color="grey", alpha=0.25)
        
        # Plot scatter points colored and filled by `stim`
        sns.scatterplot(data=group_data, x=group_data['d_phy_p'] + 10, y="s_plus", hue="stim", 
                        style="stim", palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"}, 
                        edgecolor="black", s=10, ax=ax, alpha=0.5)
        sns.scatterplot(data=group_data, x=-group_data['d_per_p'] - 10, y="s_plus", hue="stim", 
                        style="stim", palette={"CS+": "#e41a1c", "CS-": "#377eb8", "TS": "grey"}, 
                        edgecolor="black", s=10, ax=ax, alpha=0.5)

        # Add vertical line at x=0
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(group)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])

    # Customize x-axis limits and ticks
    g.set(xlim=(-max_d, max_d))
    x_ticks = [-max_d, -max_d/2, -10, 10, max_d/2, max_d]
    x_labels = [round(max_d)+10, round(max_d)/2+10, 0, 0, round(max_d)/2-10, round(max_d)-10]
    for ax in g.axes.flat:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    # Customize labels and title
    g.set_axis_labels("Distance to CS+ (left: perceptual; right: physical)", "Similarity to CS+")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Similarity")

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = ["CS+", "CS-", "TS"]
    g.add_legend(title="", handles=handles[:3], labels=legend_labels, bbox_to_anchor=(1, 0.5), loc="center right", frameon=False)

    # Save the plot
    plt.savefig(f"{plts_folder}Sim_Similarity_plus.png", dpi=300, bbox_inches="tight")
    plt.show()


plt2 = do_sim_similarity_plt()


# ## 4.4 Generalization
# 


def do_generalization_all_plt():
    G_all_data = result[result['trials'] > 24].copy()

    # Group by 'stimulus' and calculate mean of 'y' for each group
    G_all_data['mean_res'] = G_all_data.groupby('stimulus')['y'].transform('mean')

    # Group by 'participant' and 'stimulus' to calculate individual mean 'y' for each group
    G_all_data['indi_res'] = G_all_data.groupby(['participant', 'stimulus'])['y'].transform('mean')

    # Plot
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=G_all_data, x='stimulus', y='indi_res', hue='participant',
                palette=["grey"], alpha=0.3, linewidth=1, legend=None)
    sns.scatterplot(data=G_all_data, x='stimulus', y='indi_res', hue='participant',
                    palette=["grey"], alpha=0.3, s=10, legend=None)
    sns.lineplot(data=G_all_data, x='stimulus', y='mean_res', color="black", linewidth=1)
    sns.scatterplot(data=G_all_data, x='stimulus', y='mean_res', color="black", s=50)

    # Customize plot
    plt.xlabel("Stimulus")
    plt.ylabel("US Expectancy")
    plt.title("Generalized response (whole dataset)")
    plt.yticks([1, 5, 10])
    plt.ylim(0, max(G_all_data['mean_res']) + 2)
    plt.grid(True)
    plt.savefig(f"{plts_folder}Sim_Generalization_all.png", dpi=300, bbox_inches="tight")

    return plt
plt3 = do_generalization_all_plt()


def do_sim_generalization_plt():

    # Filter rows where trials > 24
    G_data = result[result['trials'] > 24].copy()

    # Group by 'group' and 'stimulus' and calculate mean of 'y' for each group
    G_data['mean_res'] = G_data.groupby(['group', 'stimulus'])['y'].transform('mean')

    # Group by 'participant' and 'stimulus' to calculate individual mean 'y' for each group
    G_data['indi_res'] = G_data.groupby(['participant', 'stimulus'])['y'].transform('mean')

    # Faceted Plot
    g = sns.FacetGrid(G_data, col='group', col_wrap=2, height=4, sharey=True)
    g.map_dataframe(sns.lineplot, x='stimulus', y='indi_res', color='grey', alpha=0.3, linewidth=1)
    g.map_dataframe(sns.scatterplot, x='stimulus', y='indi_res', color='grey', alpha=0.3, s=10)
    g.map_dataframe(sns.lineplot, x='stimulus', y='mean_res', color="black", linewidth=1)
    g.map_dataframe(sns.scatterplot, x='stimulus', y='mean_res', color="black", s=50)

    # Customize plot
    g.set_axis_labels("Stimulus", "US Expectancy")
    g.set_titles("{col_name}")
    g.set(yticks=[1, 5, 10])
    g.fig.suptitle("Generalized response", y=1.05)
    g.set(ylim=(0, max(G_data['mean_res']) + 2))

    # Remove legend and adjust layout
    plt.savefig(f"{plts_folder}Sim_Generalization.png", dpi=300, bbox_inches="tight")

    return plt

plt4 = do_sim_generalization_plt()


