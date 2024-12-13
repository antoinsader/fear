import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

root_folder = '../../'
convergence_plot_folder = root_folder + "Plots/py/4_convergence/"


result = [
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study1_CLG2D_t.pkl"),
    load_data_from_pickle(f"{root_folder}Data/fitting_res_py/Result_Study2_CLG2D_t.pkl")
]




# Define the number of participants and trials in each study
Nparticipant = 40
Ntrials_1 = 188
Ntrials_2 = 180
Nactrials_1 = 14
Nactrials_2 = 24


# Define some labels for plots
experiments = ["Exp.1: Simple conditioning" , "Exp.2: "]
color = ["#56B4E9", "yellow", "#CC79A7", "#009E73"]
color_gp = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]

group_nm = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers", "Unknown"]
colors = ["#CC79A7", "#F0E442", "#56B4E9", "#D55E00", "black"]
stimulus_level_study1 = ["S4", "S5", "S6", "CS+", "S8", "S9", "S10"]
stimulus_level_study2 = ["CS+", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "CS-"]
group = ["Non-Learners", "Overgeneralizers", "Physical Generalizers", "Perceptual Generalizers"]


# Define parameter names and corresponding Greek letters
paranames = ["sigma", "alpha", "lambda", "w0", "w1", "pi"]
greek = ["σ", "α", "λ", "w0", "w1", "p"]

# Placeholder for convergence plots
cg_plot = []



def histogram_plot(df, param_name, greek_letter, experiment):
    non_zero_df = df[df["Value"] > 1e-9]
    if non_zero_df.empty:
        print(f"Warning: No valid data for {param_name} after filtering zero or near-zero values.")
        return
    
    
    data_range = non_zero_df["Value"].max() - non_zero_df["Value"].min()
    
    # Safeguard for cases with very small ranges
    binwidth = max(data_range / 50, 1e-5)  # Fallback to a small positive binwidth
    
    try:
        g = sns.FacetGrid(non_zero_df, col="Parameter", hue="Chain", sharex=False, sharey=True, col_wrap=5, palette=color)
        g.map(sns.histplot, "Value", kde=False, binwidth=binwidth)
        g.add_legend()
        g.set_titles("{col_name}")
        g.fig.suptitle(f"{experiment} - {greek_letter}")
        g.set_axis_labels("Value", "Count")
        plt.subplots_adjust(top=0.9)

        # Save the plot
        g.savefig(convergence_plot_folder + f"{experiment}_{param_name}_histo.png", bbox_inches="tight", dpi=300)
        plt.show()

    except Exception as e:
        print(f"Error while plotting {param_name}: {e}")
        
    
    
def trace_plot(df, param_name, greek_letter, experiment):
    # Map `Iteration` as the x-axis
    g = sns.FacetGrid(df, col="Parameter", hue="Chain", sharex=False, sharey=True, col_wrap=5, palette=color)
    g.map(sns.lineplot, "Iteration", "Value", alpha=0.8)
    g.add_legend()
    g.set_titles("{col_name}")
    g.fig.suptitle(f"{experiment} - {greek_letter}")
    g.set_axis_labels("Iteration", "Value")
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    g.savefig(convergence_plot_folder + f"{experiment}_{param_name}_trace.png", bbox_inches="tight", dpi=300)
    plt.show()
    



for x, experiment in enumerate(experiments):
    for y, paraname in enumerate(paranames):       
        count = [2, 40, 40, 40, 40, 4][y] 
        para = [f"{paraname}[{i}]" for i in range(1, count + 1)]
        
        
        posterior_data = result[x]
        coords = posterior_data["coords"]
        params = coords["param"]
        samples = np.array(posterior_data["posterior"])

        
        
        indices = [i for i, name in enumerate(params) if name in para] 
        para = [params[i] for i in indices]
        parameter_samples = samples[:, :, indices]
        flattened_samples = parameter_samples.reshape(-1, len(indices))

        df = pd.DataFrame(flattened_samples, columns=para)
        df["Chain"] = np.repeat(np.arange(1, samples.shape[0] + 1), samples.shape[1])
        df["Iteration"] = np.tile(np.arange(1, samples.shape[1] + 1), samples.shape[0])  # Add Iteration column

        df = pd.melt(df, id_vars=["Chain", "Iteration"], var_name="Parameter", value_name="Value")

        trace_plot(df, paraname, greek[y], experiment)
        histogram_plot(df, paraname, greek[y], experiment)
        
paraname_2 = ["alpha_mu", "lambda_mu"]
greek_2 = ["α_μ", "λ_μ"]


def trace_plot_2(df, param_name, greek_letter, experiment):
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Iteration", y="Value", hue="Chain", palette=color, alpha=0.8)
        plt.title(f"{experiment} - {greek_letter}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend(title="Chain")
        plt.savefig(convergence_plot_folder + f"{experiment}_{param_name}_trace.png", bbox_inches="tight", dpi=300)
        
        plt.show()
    except Exception as e:
        print(f"Error while plotting trace for {param_name}: {e}")

def histogram_plot_2(df, param_name, greek_letter, experiment):
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="Value", hue="Chain", palette=color, binwidth=0.05, kde=False)
        plt.title(f"{experiment} - {greek_letter}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.legend(title="Chain")
        plt.savefig(convergence_plot_folder + f"{experiment}_{param_name}_histo.png", bbox_inches="tight", dpi=300)
        
        plt.show()
    except Exception as e:
        print(f"Error while plotting histogram for {param_name}: {e}")




for x, experiment in enumerate(experiments):
    for y, paraname in enumerate(paraname_2):
        
        
        posterior_data = result[x]
        samples = np.array(posterior_data["posterior"])
        coords = posterior_data["coords"]
        params = coords["param"]

        values = samples[:, :, params.index(paraname)].reshape(-1)
        chain = np.repeat(np.arange(1, 5), 2500)  # Assuming 4 chains, 2500 iterations each
        iteration = np.tile(np.arange(1, 2501), 4)  # 2500 iterations per chain

        estimation = pd.DataFrame({
            "Value": values,
            "Chain": chain.astype(str),
            "Iteration": iteration
        })
        

        if estimation.empty:
            print(f"Warning: DataFrame for {paraname} is empty. Skipping...")
            continue
        
        # Generate and save trace and histogram plots
        trace_plot_2(estimation, paraname, greek_2[y], experiment)
        histogram_plot_2(estimation, paraname, greek_2[y], experiment)