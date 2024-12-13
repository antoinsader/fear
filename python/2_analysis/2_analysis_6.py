#%%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



root_folder = "../../"
plots_posterior_folder= root_folder + "Plots/py/5_posterior_predictive/"



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




def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

pre_data = load_data_from_pickle(f'{root_folder}Data/res_py/2_an_predata.pkl')





_pre_data_cut = [
    pre_data[a][0] for a in range(2)
]

pre_data_cut = [
    pd.DataFrame(_pre_data_cut[0]),
    pd.DataFrame(_pre_data_cut[1])
]

#%%

_pre_data_cut[0]
#%%

ppc_plot_data = [
    pd.DataFrame({
        'stimulus': pd.concat([pre_data_cut[a]['stimulus']] * 6, ignore_index=True),
        'sim_y': pd.concat([
            pre_data_cut[a]['s_mean'],
            pre_data_cut[a]['s_q10'],
            pre_data_cut[a]['s_q30'],
            pre_data_cut[a]['s_q50'],
            pre_data_cut[a]['s_q70'],
            pre_data_cut[a]['s_q90']
        ], ignore_index=True),
        'act_y': pd.concat([
            pre_data_cut[a]['mean'],
            pre_data_cut[a]['q10'],
            pre_data_cut[a]['q30'],
            pre_data_cut[a]['q50'],
            pre_data_cut[a]['q70'],
            pre_data_cut[a]['q90']
        ], ignore_index=True),
        'quantile': pd.concat([
            pd.Series(["Mean"] * len(pre_data_cut[a]['stimulus'])),
            pd.Series(["10% quantile"] * len(pre_data_cut[a]['stimulus'])),
            pd.Series(["30% quantile"] * len(pre_data_cut[a]['stimulus'])),
            pd.Series(["50% quantile"] * len(pre_data_cut[a]['stimulus'])),
            pd.Series(["70% quantile"] * len(pre_data_cut[a]['stimulus'])),
            pd.Series(["90% quantile"] * len(pre_data_cut[a]['stimulus']))
        ], ignore_index=True)
    })
    for a in range(2)
]



#%%

titles = ["Exp.1: Simple conditioning", "Exp.2: Differential conditioning"]

fig, axes = plt.subplots(1, 2, figsize=(16.5, 12), sharey=True)
for a in range(2):
    sns.lineplot(
        data=ppc_plot_data[a], 
        x='stimulus', 
        y='act_y', 
        hue='quantile', 
        ax=axes[a], 
        label='Observed data', 
        color='black'
    )
    sns.lineplot(
        data=ppc_plot_data[a], 
        x='stimulus', 
        y='sim_y', 
        hue='quantile', 
        ax=axes[a], 
        label='Model-based prediction', 
        color='blue'
    )
    axes[a].set_title(titles[a])
    axes[a].set_ylim(-2, 11.5)
    axes[a].set_xticks([1, 10, 5])
    axes[a].legend(loc='lower right')

plt.tight_layout()
plt.savefig(plots_posterior_folder + f"PPC.png", dpi=300)
plt.show()

#%%

# ppc_plot_gr_data = [
#     pre_data[a][1][pre_data[a][1]['category'] != 'Unknown'] for a in range(2)
# ]
# group_nm = group
# fig, axes = plt.subplots(2, 1, figsize=(18.2, 14), sharey=True)

# for a in range(2):
#     data = ppc_plot_gr_data[a]
#     g = sns.FacetGrid(data, col="category", col_order=group_nm, col_wrap=5, sharey=True, height=5, aspect=1.5)
#     g.map(plt.plot, "stimulus", "q10", color="black", label="Observed q10", linestyle="-")
#     g.map(plt.scatter, "stimulus", "q10", color="black", label="Observed q10", s=50, marker='s')
#     g.map(plt.plot, "stimulus", "s_q10", color="blue", label="Simulated q10", linestyle="--")
#     g.map(plt.scatter, "stimulus", "s_q10", color="blue", label="Simulated q10", s=50, marker='s')

#     g.map(plt.plot, "stimulus", "q50", color="black", label="Observed q50", linestyle="-")
#     g.map(plt.scatter, "stimulus", "q50", color="black", label="Observed q50", s=50, marker='o')
#     g.map(plt.plot, "stimulus", "s_q50", color="blue", label="Simulated q50", linestyle="--")
#     g.map(plt.scatter, "stimulus", "s_q50", color="blue", label="Simulated q50", s=50, marker='o')

#     g.map(plt.plot, "stimulus", "q90", color="black", label="Observed q90", linestyle="-")
#     g.map(plt.scatter, "stimulus", "q90", color="black", label="Observed q90", s=50, marker='^')
#     g.map(plt.plot, "stimulus", "s_q90", color="blue", label="Simulated q90", linestyle="--")
#     g.map(plt.scatter, "stimulus", "s_q90", color="blue", label="Simulated q90", s=50, marker='^')

#     g.set_titles(col_template="{col_name}")
#     g.set_axis_labels("Stimulus", "US Expectancy (Observed Data Scale: 1-10)")
#     g.set(ylim=(-2, 11.5))
#     g.add_legend()

#     axes[a].set_title(titles[a], fontsize=16)
    
# plt.tight_layout()
# fig.suptitle("Posterior Predictive Checks", fontsize=18, fontweight="bold", y=1.02)
# plt.savefig(plots_posterior_folder + f"ppc_gr.png", dpi=300)
# plt.show()

# ppc_plot_indi_data = [
#     pre_data[x][2] for x in range(len(experiments))  # Extract individual-level data
# ]

# for x in range(len(experiments)):
#     data = ppc_plot_indi_data[x]
#     g = sns.FacetGrid(
#         data,
#         col="participant",
#         col_wrap=5,
#         sharey=True,
#         height=4,
#         aspect=1.2
#     )
#     g.map(plt.plot, "stimulus", "mean", color="black", label="Observed Mean")
#     g.map(plt.plot, "stimulus", "s_mean", color="blue", linestyle="--", label="Simulated Mean")
#     g.map(plt.scatter, "stimulus", "mean", color="black", s=50)
#     g.map(plt.scatter, "stimulus", "s_mean", color="blue", s=50)

#     # Add category labels
#     for ax, participant in zip(g.axes.flat, data["participant"].unique()):
#         subset = data[data["participant"] == participant]
#         category = subset["category"].iloc[0]
#         ax.text(
#             4 if x == 0 else 5, 11, category,
#             fontsize=10, ha="center", va="top", bbox=dict(facecolor="gray", alpha=0.5)
#         )

#     g.set_titles("{col_name}")
#     g.set_axis_labels("Stimulus", "US Expectancy (Observed Data Scale: 1-10)")
#     g.set(ylim=(-1, 11.5), xticks=[1, 5, 10])
#     g.fig.subplots_adjust(top=0.9)
#     g.fig.suptitle(experiments[x], fontsize=16, fontweight="bold")

#     # Save the plot
#     plt.savefig(plots_posterior_folder + f"PPC_plot_indi{x+1}.png", dpi=300, bbox_inches="tight")

#     plt.close(g.fig)
# %%
