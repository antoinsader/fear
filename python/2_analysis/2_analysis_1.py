import pandas as pd
import numpy as np
import pickle
import pymc as pm
import pytensor.tensor as at
import pytensor
from pytensor.tensor.extra_ops import broadcast_to







def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
    
d_list = {
    "e1_clg": load_data_from_pickle("../Data/res_py/Data1_JAGSinput_CLG.pkl"),
}

data = d_list['e1_clg']

Nparticipants = data['Nparticipants']
Ntrials = data['Ntrials']
Nactrials = data['Nactrials']

d_per = data['d_per']
d_phy = data['d_phy']


d_per = (d_per - np.mean(d_per)) / np.std(d_per)
d_phy = (d_phy - np.mean(d_phy)) / np.std(d_phy)



y = data['y']
r = data['r']
k = data['k']
A = 1
K = 10


with pm.Model() as model:
    
    pi = pm.Dirichlet("pi", a=np.ones(4) , shape=4)
    gp = pm.Categorical("gp", p=pi, shape=Nparticipants)

    # Priors 
    lambda_mu = pm.TruncatedNormal("lambda_mu", mu=0.1, sigma=1, lower=0)

    lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
    
    
    alpha_raw = pm.Beta("alpha_raw", alpha=1, beta=1)
    alpha_mu = pm.Deterministic("alpha_mu", pm.math.clip(alpha_raw, 1e-9, 1 - 1e-9))
    
    alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
    alpha_a = alpha_mu * alpha_kappa
    alpha_b = (1 - alpha_mu) * alpha_kappa
    
    w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
    w0_sigma = pm.Gamma("w0_sigma", alpha=2, beta=1)
    
    w1_a_raw = pm.Gamma("w1_a_raw", alpha=2, beta=1)
    w1_b_raw = pm.Gamma("w1_b_raw", alpha=2, beta=1)
    w1_a = pm.Deterministic("w1_a", pm.math.maximum(w1_a_raw, 1e-9))
    w1_b = pm.Deterministic("w1_b", pm.math.maximum(w1_b_raw, 1e-9))
    
    
    
    sigma1 = pm.Uniform("sigma1", lower=1e-9, upper=1.5)  # Noise for learners
    sigma2 = pm.Uniform("sigma2", lower=1.5, upper=3)    # Noise for non-learners
    sigma1_broadcasted = at.full((Nparticipants, Ntrials), sigma1)
    sigma2_broadcasted = at.full((Nparticipants, Ntrials), sigma2)

    
    
    
    # --- Participant-level Parameters ---
    w0 = pm.Normal("w0", mu=w0_mu, sigma=w0_sigma, shape=Nparticipants)
    
    w1_1_raw = pm.Gamma("w1_1_raw", alpha=w1_a, beta=w1_b, shape=Nparticipants)
    w1_1 = pm.Deterministic("w1_1", pm.math.maximum(w1_1_raw, 1)) 
    w1 = pm.Deterministic("w1", pm.math.switch( at.eq(gp, 1), 0, w1_1))
    
    
    alpha_1_raw = pm.Beta("alpha_1_raw", alpha=alpha_a, beta=alpha_b, shape=Nparticipants)
    alpha_1 = pm.Deterministic("alpha_1", pm.math.clip(alpha_1_raw, 1e-9, 1 - 1e-9))  # Apply truncation to enforce bounds
    alpha = pm.Deterministic("alpha", pm.math.switch(at.eq(gp, 1), 0, alpha_1))  
    
    
        

    lambda_1_raw = pm.Normal("lambda_1_raw", mu=lambda_mu, sigma=lambda_sigma, shape=Nparticipants)
    lambda_1 = pm.Deterministic("lambda_1", pm.math.maximum(lambda_1_raw, 0.0052))
    
    lambda_2_raw = pm.Normal("lambda_2_raw", mu=lambda_mu, sigma=lambda_sigma, shape=Nparticipants)
    lambda_2 = pm.Deterministic("lambda_2", pm.math.minimum(lambda_2_raw, 0.0052))
    
    
    
    lambda_ = pm.Deterministic(
        "lambda",
        at.switch(
            at.eq(gp, 1),
            0,
            at.switch(at.eq(gp,2), lambda_2, lambda_1)
        )
    )
    
    
    d_sigma_raw = pm.Gamma("d_sigma_raw", alpha=2, beta=1, shape=Nparticipants)
    d_sigma = pm.Deterministic("d_sigma", pm.math.maximum(d_sigma_raw, 1e-9))
    
    
    zn = pm.math.switch(at.eq(gp, 1), 2, 1)
    zn = at.reshape(zn, (Nparticipants,))
    zn_broadcasted = at.broadcast_to(zn[:, None], (Nparticipants, Ntrials))

    

    gp_broadcasted = at.broadcast_to((gp > 1)[:, None], d_per.shape)  
    s = pm.Deterministic("s", pm.math.switch(
        (gp_broadcasted & (d_per > 0)),
        pm.math.exp(-at.broadcast_to(lambda_[:, None], d_per.shape) * pm.math.switch(at.eq(at.broadcast_to(gp[:, None], d_per.shape), 4), d_per, d_phy)),
        1
    ))
    

    k_t = at.as_tensor_variable(k.T)
    r_t = at.as_tensor_variable(r.T)
    alpha_sym = at.as_tensor_variable(alpha)
    gp_sym = at.as_tensor_variable(gp)
    v0 = at.zeros((Nparticipants,))  
    
    def v_update(v_prev, r_t, k_t, gp_i,alpha_i):
        v_next = at.switch(
            at.neq(gp_i, 1),
            at.switch(  at.eq( k_t, 1), v_prev + alpha_i * (r_t - v_prev), v_prev),
            0
        )
        return v_next

    v_updates, _ = pytensor.scan(
        fn=v_update,
        sequences=[r_t, k_t],
        outputs_info=v0,
        non_sequences=[gp_sym, alpha_sym]
    )
    
    v_updates_t = v_updates.T
    v = pm.Deterministic("v", v_updates_t)
    g = pm.Deterministic("g", v * s)
   
   
   
    w0_broadcasted = at.broadcast_to(w0[:, None], g.shape) 
    w1_broadcasted = at.broadcast_to(w1[:, None], g.shape) 
   
    theta = pm.Deterministic(
        "theta",
        A + (K - A) / (1 + pm.math.exp(-(w0_broadcasted + w1_broadcasted * g)))
    )
    
    zn_eq_1 = at.eq(zn_broadcasted, 1)
    sigma_broadcasted = at.switch(
        at.broadcast_to(zn_eq_1, sigma1_broadcasted.shape),
        sigma1_broadcasted,
        sigma2_broadcasted
    )
        
    
    
    
    # Observed data
    y_obs = pm.Normal("y_obs", mu=theta, sigma=sigma_broadcasted, observed=y)

        
    intermediate_switch = at.switch(zn_eq_1, sigma1_broadcasted, sigma2_broadcasted)
    
   
    y_pre = pm.Normal("y_pre",mu=theta[:, Nactrials:],sigma=sigma_broadcasted[:, Nactrials:],shape=(Nparticipants, Ntrials - Nactrials))
    

    
    loglik = pm.logp(pm.Normal.dist(mu=theta[:, Nactrials:], sigma=sigma_broadcasted[:, Nactrials:]), y[:, Nactrials:])
    
    
    print("Shape of theta[:, Nactrials:]:", theta[:, Nactrials:].shape)
    print("Shape of sigma_broadcasted[:, Nactrials:]:", sigma_broadcasted[:, Nactrials:].shape)
    print("Shape of y[:, Nactrials:]:", y[:, Nactrials:].shape)

    
    trace = pm.sample(
        draws=25000,       
        tune=75000,        
        chains=4,          
        cores=4,           
        target_accept=0.9, 
        return_inferencedata=True  
    )
    with open('./fitting_res_py/Result_Study1_CLG2D_trace.pkl', 'wb') as f:
        pm.save_trace(trace, f)


