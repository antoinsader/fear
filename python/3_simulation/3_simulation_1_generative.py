import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


root_folder = "../../"
plts_folder = root_folder + "Plots/py/2_SimulationPlots/"


# Generate a sequence of stimulus sizes ranging from 50.80 to 119.42 in increments of 7.624, then round to 2 decimal places
size = np.round(np.arange(50.80, 119.42 + 7.624, 7.624), 2)
# Set the CS+ stimulus size to the 5th value in the 'size' sequence
CSp_size = size[4]
# Set the number of learning trials (Nactrials) and generalization trials (Ngetrials)
Nactrials = 24
Ngetrials = 76
Ntrials = Nactrials + Ngetrials



np.random.seed(20221001)
# Create a vector 's_size' with the CS+ stimulus size repeated for the number of learning trials, and a random selection of sizes from the 'size' sequence for the number of generalization trials

s_size = np.concatenate((np.repeat(CSp_size, Nactrials), np.random.choice(size, Ngetrials, replace=True)))

# Calculate the absolute difference between each value in 's_size' and the CS+ stimulus size, then round to the nearest integer
d = np.abs(np.round(s_size - CSp_size, 0))

# Create a vector 'stimulus' with the values "CS+" if the corresponding value in 'd' is 0, and "TS" otherwise
stimulus = np.where(d == 0, "CS+", "TS")

# Create a vector 'r' with a random selection of 1s and 0s with a probability of 0.8 and 0.2, respectively, and a length equal to the number of trials
# Randomly assign response (1 or 0) for each trial with probabilities 0.8 and 0.2
r = np.random.choice([1, 0], size=len(s_size), p=[0.8, 0.2])

# Create a vector 'k' with the value 1 if the corresponding value in 'd' is 0, and 1 otherwise
k = np.where(d == 0, 1, 0)

# ### Model:
# 

def lg_fun (alpha, _lambda, w1 =10, w0 =-5, sigma = 0,  trials = Ntrials,  K=10, A=1, persd=0):
    
    y = np.zeros(trials)
    theta = np.zeros(trials)
    s_per = np.zeros(trials)
    s_phy = np.zeros(trials)
    perd = np.zeros(trials)
    phyd = d 
    v = np.zeros(trials + 1)
    g = np.zeros(trials)
    perd = np.zeros(trials)
    color = ["red", "#56B4E9"]

    
    for t in range(trials):
        perd[t] = abs(np.random.normal(phyd[t], persd))
        v[t + 1] = v[t] + alpha * (r[t] - v[t]) if k[t] == 1 else v[t]
        s_per[t] = np.exp(-_lambda * perd[t])
        s_phy[t] = np.exp(-_lambda * phyd[t])
        g[t] = v[t] * s_per[t]
        theta[t] = A + (K - A) / (1 + np.exp(-(w0 + w1 * g[t])))
        y[t]  = np.random.normal(theta[t], sigma)

    data = pd.DataFrame({
        'theta': theta,
        's_per': s_per,
        's_phy': s_phy,
        'v': v[:Ntrials],  
        'g': g,
        'stim': stimulus,  
        'y': y,
        'trials': np.arange(1, Ntrials + 1),  
        'phyd': phyd,
        'perd': perd,
        'r': r,
        's_size': s_size
    })
    data['percept'] = np.where(data['perd'] < 5, '1', 
                           np.where((data['perd'] > 5) & (data['perd'] < 15), '2', '3'))
    return data


def do_plt1(data, title, alpha, _lambda, persd = 0):
    stim_colors = {'CS+': 'red', 'TS': '#56B4E9'}

    plot_data = data[data['trials'] <= 24]
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['trials'], plot_data['v'], linestyle='-', linewidth=1.2, color='black', label='v')
    shape_map = {1: 'o', 0: 's'}
    for i, row in plot_data.iterrows():
        plt.scatter(row['trials'], row['v'], color=stim_colors[row['stim']], s=100, marker=shape_map[row['r']])

    plt.ylim(0, 1)
    plt.xlabel('Trials')
    plt.ylabel('Associative strengths')
    plt.title(f"{title}\nα = {alpha} ; λ = {_lambda}; perceptual sd = {persd}")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('white')
    plt.legend(loc='lower center', ncol=2, frameon=False)
    plt1 = plt    
    return plt1 
def do_plt2(data, title, alpha, _lambda, persd = 0):
    stim_colors = {'CS+': 'red', 'TS': '#56B4E9'}

    # Plot 2: Similarity to CS+ based on stimulus size    
    data_plt2 = data.sort_values(by='s_size')
    plt.figure(figsize=(10, 6))
    plt.plot(data_plt2['s_size'], data_plt2['s_phy'], linestyle='-', linewidth=1.2, color='black', label="Physical similarity to CS+")
    plt.scatter(data_plt2['s_size'], data_plt2['s_per'], color=[stim_colors[stim] for stim in data_plt2['stim']], s=100, label="Perceived similarity")
    
    plt.xlabel('Stimulus size')
    plt.ylabel('Similarity to CS+')
    plt.title(f"{title}\nα = {alpha} ; λ = {_lambda}; perceptual sd = {persd}")

    plt.legend(["Physical similarity to CS+", "Perceived similarity"], loc='lower center', ncol=2, frameon=False)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('white')
    plt2 = plt    
    return plt2
def do_plt3(data, title, alpha,_lambda, persd = 0):
    data = data.sort_values(by='s_size')
    data['percept'] = np.where(data['perd'] < 10, '1', np.where((data['perd'] > 10) & (data['perd'] < 20), '2', '3'))
    data['meansper'] = data.groupby('s_size')['s_per'].transform('mean')
    plt.figure(figsize=(10, 6))
    plt.plot(data['s_size'], data['s_phy'], linestyle='--', color='grey', linewidth=1.2, label="Physical similarity")
    unique_sizes = data['s_size'].unique()
    mean_per_size = data.groupby('s_size')['meansper'].mean()
    plt.plot(unique_sizes, mean_per_size, color='black', linewidth=1.2, label="Mean perceived similarity")
    stim_colors = {'CS+': 'red', 'TS': '#56B4E9'}
    shape_map = {'1': 'o', '2': '^', '3': 's'}  

    for percept in data['percept'].unique():
        subset = data[data['percept'] == percept]
        plt.scatter(subset['s_size'], subset['s_per'], 
                    color=[stim_colors[stim] for stim in subset['stim']], 
                    edgecolor='black', s=100, 
                    marker=shape_map[percept], label=f'Perceptual distance {percept}')

    plt.xlabel('Stimulus size')
    plt.ylabel('Similarity to CS+')
    plt.ylim(0, 1)
    plt.xticks(sorted(data['s_size'].unique()))
    plt.yticks(np.arange(0, 1.25, 0.25))
    plt.title(f"{title}\nα = {alpha} ; λ = {_lambda}; perceptual sd = {persd}")

    # Add legend and grid
    plt.legend(title="Legend", loc='lower center', ncol=3, frameon=False)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('white')
    plt3 = plt
    return plt3


K=10
A=1
w1=10
w0=-5
trials = 100

alpha_high = 0.3
alpha_low = 0.01
lambda_ = 0.3

L_sim_high = lg_fun(alpha_high, lambda_, w1=10)
L_sim_high_plot = do_plt1(L_sim_high, "High learning rate",alpha_high,lambda_ )
L_sim_high_plot.savefig(f"{plts_folder}/ChangeLearning_High.png", dpi=300)

L_sim_low = lg_fun(alpha_low, lambda_, w1=10)
L_sim_low_plot = do_plt1(L_sim_low, "Low learning rate",alpha_low,lambda_ )
L_sim_low_plot.savefig(f"{plts_folder}/ChangeLearning_Low.png", dpi=300)





# ### Alter generalization rate:

alpha = 0.1
lambda_high = 0.3
lambda_low = 0.01

G_sim_high = lg_fun(alpha, lambda_high, w1=10)
G_sim_high_plot = do_plt2(G_sim_high, "High generalization rate ",alpha_high,lambda_ )
G_sim_high_plot.savefig(f"{plts_folder}/ChangeGeneralization_High.png", dpi=300)

G_sim_low = lg_fun(alpha, lambda_low, w1=10)
G_sim_low_plot = do_plt2(G_sim_low, "Low generalization rate",alpha_low,lambda_ )
G_sim_low_plot.savefig(f"{plts_folder}/ChangeGeneralization_Low.png", dpi=300)



# ### Alter persd

pers_d_low = 2
pers_d_high = 20

P_sim_high = lg_fun(0.1, 0.1, w1=10, persd=pers_d_high )
P_sim_high_plot = do_plt3(P_sim_high, "High perceptual variability",0.1,0.1 )
P_sim_high_plot.savefig(f"{plts_folder}/ChangePersd_High.png", dpi=300)

P_sim_low = lg_fun(0.1,0.1, w1=10, persd=pers_d_low)
P_sim_low_plot = do_plt3(P_sim_low, "Low perceptual variability",0.1,0.1 )
P_sim_low_plot.savefig(f"{plts_folder}/ChangePersd_Low.png", dpi=300)



def logit(w01, w02, w03, w04, w05, w06, w11, w12, w13, w14, w15, w16):
    # Generate a grid of g values (less dense for clarity)
    g = np.linspace(-1, 1, 10)  # Using fewer points for better clarity
    p_matrices = []

    # Calculate probabilities for each pair of w0 and w1
    weights = [(w01, w11), (w02, w12), (w03, w13), (w04, w14), (w05, w15), (w06, w16)]
    for w0, w1 in weights:
        p = 1 + 9 / (1 + np.exp(-(w0 + w1 * g)))  # Compute θ values
        p_matrices.append(p)

    # Plot all probabilities with clearer style
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "pink", "green", "purple", "brown"]
    labels = [
        f"w0 = {w01}, w1 = {w11}", f"w0 = {w02}, w1 = {w12}", f"w0 = {w03}, w1 = {w13}",
        f"w0 = {w04}, w1 = {w14}", f"w0 = {w05}, w1 = {w15}", f"w0 = {w06}, w1 = {w16}"
    ]

    # Loop through each set of probabilities, using clean lines and small markers
    for p, color, label in zip(p_matrices, colors, labels):
        plt.plot(g, p, color=color, marker='o', markersize=5, label=label, linestyle='-', linewidth=1.5)

    plt.xlabel("g")
    plt.ylabel("θ")
    plt.ylim(1, 10)
    plt.legend(title="Parameter pairs", loc='upper left', fontsize=9)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(f"{plts_folder}/ChangeW.png", dpi=300)
    plt.show()

# Test the function
logit(w01=3, w02=3, w03=0, w04=0, w05=-3, w06=-3, w11=5, w12=20, w13=5, w14=20, w15=5, w16=20)


