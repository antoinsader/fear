import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator



data_s1 = pd.read_csv("../../Data/res_py/data_s1_py.csv")
data_s2 = pd.read_csv('../../Data/res_py/data_s2_py.csv')

Nactrials_1 = 14
Nactrials_2 = 24
Ngetrials_1 = 174
Ngetrials_2 = 156
Ntrials_1 = Nactrials_1 + Ngetrials_1
Ntrials_2 = Nactrials_2 + Ngetrials_2


# ## 3. Visuilization
# ### Prepare for visualization:

data_s1_dropped =data_s1.dropna(axis='index', how='all', subset=['US_expect'])
data_s2_dropped =data_s2.dropna(axis='index', how='all', subset=['US_expect'])
data_list = [data_s1_dropped, data_s2_dropped]
Nactrials = [Nactrials_1, Nactrials_2]  # Assuming these values are defined
plot_list = []
colors = ['red', 'blue'] 
root_folder = "../../"
plts_folder = root_folder + "Plots/py/1_DataCleanPlots/"



# ### Learning:
# #### Group


for i, data in enumerate(data_list):
    # Filter data to include only the first Nactrials[i] trials
    filtered_data = data[data['trials'] <= Nactrials[i]]
    
    # Group by 'trials' and 'stimulus', and calculate mean and standard deviation
    summarized_data = filtered_data.groupby(['trials', 'stimulus'], observed=False).agg(

        mean_ac=('US_expect', 'mean'),
        sd_ac=('US_expect', 'std')
    ).reset_index()
    
    # Merge back with filtered_data to calculate individual participant metrics
    summarized_data = pd.merge(filtered_data, summarized_data, on=['trials', 'stimulus'])
    summarized_data['mean_ac_indi'] = summarized_data.groupby(['trials', 'stimulus', 'participant'],observed=False)['US_expect'].transform('mean')
    summarized_data['sd_ac'] = summarized_data.groupby(['trials', 'stimulus', 'participant'],observed=False)['US_expect'].transform('std')

    # Plotting
    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(data=summarized_data, x='trials', y='mean_ac', hue='stimulus', style='stimulus', markers=True, err_style="bars", errorbar='sd',  palette=colors)
    
    # Adding individual participant lines
    for key, grp in summarized_data.groupby(['participant', 'stimulus'], observed=False):
        ax = grp.plot(ax=ax, x='trials', y='mean_ac_indi', label='_nolegend_', color='grey', alpha=0.2)
    
    # Setting plot title and labels
    ax.set_title(f"Acquisition: Exp.{i+1}")
    ax.set_xlabel("Trials")
    ax.set_ylabel("US expectancy (1 - 10)")
    
    # Setting axis limits and ticks
    ax.set_ylim(0, 10.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.legend(title='Stimulus', labels=['CS+', 'CS-'], title_fontsize='13', fontsize='11', loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{plts_folder}1_Acquisition_Exp_{i+1}.jpg", format='jpg', dpi=300)
    # Store the plot in the list
    plot_list.append(ax.get_figure())



# Individual:


for i, data in enumerate(data_list):
    # Filter data to include only the first Nactrials[i] trials
    filtered_data = data[data['trials'] <= Nactrials[i]]
    
    # Calculate mean of US_expect by participant, trial, and stimulus
    summarized_data = filtered_data.groupby(['participant', 'trials', 'stimulus'],observed=False).agg(
        mean_ac=('US_expect', 'mean')
    ).reset_index()
    
    # Plotting with Seaborn and Matplotlib
    g = sns.FacetGrid(summarized_data, col='participant', col_wrap=8, hue='stimulus', height=2.5, aspect=1)
    g.map_dataframe(sns.lineplot, x='trials', y='mean_ac')
    g.map_dataframe(sns.scatterplot, x='trials', y='mean_ac', s=30)
    
    # Customizing plots
    g.set_titles("Participant {col_name}")
    g.set_axis_labels("Trials", "US Expectancy")
    g.add_legend(title="Stimulus")
    for ax in g.axes.flatten():
        ax.set_title(ax.get_title(), fontsize='small')
        ax.set_ylim(0, 10.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensures integer values on x-axis
    
    # Saving each plot in the list
    g.savefig(f"{plts_folder}2_Experiment_{i+1}.jpg", format='jpg', dpi=300)
    
    plot_list.append(g)
    g.figure.suptitle(f"Experiment {i+1}", y=1.02)
    g.figure.tight_layout()
    g.figure.show()




# ### Generalization:


# Group:



def create_generalization(data, trials_threshold, title, stimulus_order, imgTitle):
    filtered_data = data[data['trials'] > trials_threshold]
    filtered_data['stimulus'] = pd.Categorical(filtered_data['stimulus'], categories=stimulus_order, ordered=True)

    # Calculate mean and standard deviation of US_expect by stimulus and participant
    summarized_data_indi = filtered_data.groupby(['stimulus', 'participant'], as_index=False, observed=False).agg(
        mean_ge_indi=pd.NamedAgg(column='US_expect', aggfunc='mean'),
        sd_ge_indi=pd.NamedAgg(column='US_expect', aggfunc='std')
    )
    
    # Calculate mean and standard deviation of US_expect by stimulus
    summarized_data = filtered_data.groupby('stimulus', as_index=False, observed=False).agg(
        mean_ge=pd.NamedAgg(column='US_expect', aggfunc='mean'),
        sd_ge=pd.NamedAgg(column='US_expect', aggfunc='std')
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot individual participant lines
    sns.lineplot(data=summarized_data_indi, x='stimulus', y='mean_ge_indi', hue='participant', 
                 alpha=0.2, linewidth=0.5, legend=False)
    
    # Plot summary points
    sns.scatterplot(data=summarized_data, x='stimulus', y='mean_ge', s=100, color='black')
    
    # Plot summary line
    sns.lineplot(data=summarized_data, x='stimulus', y='mean_ge', linewidth=2, color='black')
    
    # Add labels and title
    plt.xlabel("Stimulus")
    plt.ylabel("US expectancy (1 - 10)")
    plt.title(title)
    
    # Set limits and breaks for the y-axis
    plt.ylim(0, 10.5)
    plt.yticks([1, 5, 10])
    
    # Use the default theme with grid
    sns.set_theme(style="whitegrid")

    plt.savefig(f"{plts_folder}3_{imgTitle}", format='jpg', dpi=300)

    # Save figure in list
    plot_list.append(plt.gcf())  # Store the current figure object






# create_generalization(data_list[0], Nactrials[0], "Generalization: Exp.1", ['S4','S5','S6', 'CS+','S8','S9','S10'], 'group_generalization_exp1.jpg')
# create_generalization(data_list[1], Nactrials[1], "Generalization: Exp.2", ['CS+','S2','S3','S4','S5','S6', 'S7','S8','S9','CS-'], 'group_generalization_exp2.jpg')



# Individual:


def create_Individual(data, trials_threshold, title, imgTitle):
    # Filter data to include only trials after the given threshold
    filtered_data = data[data['trials'] > trials_threshold]

    
    # Calculate mean and standard deviation of US_expect by stimulus and participant
    summarized_data = filtered_data.groupby(['stimulus', 'participant'], as_index=False).agg(
        mean_ge=pd.NamedAgg(column='US_expect', aggfunc='mean'),
        sd_ge=pd.NamedAgg(column='US_expect', aggfunc='std')
    )


    # Create the plot
    g = sns.FacetGrid(summarized_data, col="participant", col_wrap=5, sharey=True, sharex=True)
    g.map_dataframe(sns.pointplot, x='stimulus', y='mean_ge' )
    g.map_dataframe(sns.lineplot, x='stimulus', y='mean_ge')
    g.map(plt.errorbar, 'stimulus', 'mean_ge', 'sd_ge', fmt='o', capsize=5, capthick=1, ecolor='black')
    
    # Add labels and title
    g.set_axis_labels("Stimulus", "US expectancy (1 - 10)")
    g.figure.suptitle(title, fontsize=16)
    g.set(ylim=(0, 11))
    plt.savefig(f"{plts_folder}4_{imgTitle}", format='jpg', dpi=300)

    plt.subplots_adjust(top=0.9)
    plot_list.append(plt.gcf())  
    plt.show()



# create_Individual(data_list[0], Nactrials_1, "Experiment 1", "individual_generalization_exp1.jpg")
# create_Individual(data_list[1], Nactrials_2, "Experiment 2", "individual_generalization_exp1.jpg")



# ### Perception:


# Group:


from matplotlib.ticker import MultipleLocator



def create_perception_group_plot(data, title, img_title):
    sns.set_theme(style="whitegrid")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = sns.violinplot(
        x="stimulus_phy", 
        y="Per_size", 
        data=data, 
        inner=None, 
        color="gray", 
        alpha=0.5
    )
    
    # Add points for individual data
    sns.stripplot(
        x="stimulus_phy", 
        y="Per_size", 
        data=data, 
        jitter=True, 
        size=2, 
        alpha=0.2, 
        color="black", 
        dodge=True
    )
    
    # Add the title and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Stimulus", fontsize=12)
    plt.ylabel("Perceived Size", fontsize=12)
    
    # Customize y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0, 200)
    plt.yticks(range(0, 201, 40))
    
    # Save the plot
    plt.savefig(f"{plts_folder}6_{img_title}", format='jpg', dpi=300)

    plt.show()




create_perception_group_plot(data_list[0], "Group Perception: Exp.1", "group_perception_exp1.jpg")
create_perception_group_plot(data_list[1], "Group Perception: Exp.2", "group_perception_exp2.jpg")


# Individual:
P_in_plot = []
titles = ["Experiment 1", "Experiment 2"]
for data in data_list:
    data['stimulus_phy'] = data['stimulus_phy'].astype('category')

P_in_plot = []
idx = 0
for x, data in enumerate(data_list):
    fig, axes = plt.subplots(5, 8, figsize=(20, 12))  # Adjust size for better spacing
    axes = axes.flatten()
    idx += 1
    for i, participant_id in enumerate(sorted(data['participant'].unique())):
        ax = axes[i]
        
        # Filter data for each participant
        participant_data = data[data['participant'] == participant_id]
        
        # Boxplot
        sns.boxplot(
            data=participant_data,
            x='stimulus_phy',
            y='Per_size',
            ax=ax,
            color="white",
            fliersize=0,  # Hide outliers
            linewidth=0.8
        )
        
        # Add red line for Phy_size
        ax.plot(
            participant_data['stimulus_phy'].cat.codes,
            participant_data['Phy_size'],
            color='red',
            linewidth=1.2
        )
        
        # Add red dots for Phy_size
        ax.scatter(
            participant_data['stimulus_phy'].cat.codes,
            participant_data['Phy_size'],
            color='red',
            edgecolor='black',
            s=30,
            zorder=5
        )
        
        # Adjust axis labels
        ax.set_title(str(participant_id), fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.tick_params(axis='both', which='major', labelsize=6)

    # Remove extra axes
    for ax in axes[len(data['participant'].unique()):]:
        ax.remove()

    # Overall title for the experiment
    fig.suptitle(titles[x], fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{plts_folder}7_group_perception_exp{idx}.jpg", format='jpg', dpi=300)
    P_in_plot.append(fig)


# Show the plots (or you can save them using fig.savefig("filename.png"))
plt.show()




