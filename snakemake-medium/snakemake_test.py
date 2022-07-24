"""
Adapted from: https://ipython-books.github.io/75-fitting-a-probability-distribution-to-data-with-the-maximum-likelihood-method/
"""
import argparse
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

#Notice that in shell we are running the Python script snakemake_test.py, which takes in df and f as command-line options/arguments. This brings us to the parser_arguments() function in the script:
def parser_arguments():
    par = argparse.ArgumentParser()
    parser = par.add_argument_group('required arguments')
    parser.add_argument("-csv", "--csv", help="directory/path to input csv file", required=True)
    parser.add_argument("-p", "--plot", help="plots data", action='store_true')
    parser.add_argument("-pe", "--plot_exp", help="plots exponential fit to data", action='store_true')
    parser.add_argument("-pbs", "--plot_bs", help="plots BS fit to data", action='store_true')
    parser.add_argument("-o", "--output_plot", help="output plot filename", required=False)
    parser.add_argument("-oe", "--output_exp", help="output exponential fit plot filename", required=False)
    parser.add_argument("-ob", "--output_bs", help="output BS fit plot filename", required=False)
    args = par.parse_args()
    return args
#required=True will result in an error if the specified option is not present in the command line. As for action='store_true' (resp. action='store_false' ), it is used for storing the value True (resp. False ) and it creates a default value of False (resp. True ). Now you may be wondering where this is actually used.

def plot_survival(data, args):
    survival = data.survival
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(sorted(survival)[::-1], 'o')
    ax1.set_xlabel('Patient')
    ax1.set_ylabel('Survival time (days)')
    
    ax2.hist(survival, bins=15)
    ax2.set_xlabel('Survival time (days)')
    ax2.set_ylabel('Number of patients')
    
    plt.savefig(args.output_plot)
#If you refer back to rule test in the Snakefile, the option -p has been specified under shell . This results in args.plot to have a value True , so in main() the function plot_survival(data, args) will be executed. 

# function to fit exponential distribution to data
def fit_exp(data, args):
    survival = data.survival
    smean = survival.mean()
    rate = 1. / smean
    smax = survival.max()
    days = np.linspace(0., smax, 1000)
    dt = smax / 999.
    dist_exp = st.expon.pdf(days, scale=1. / rate)
    
    nbins = 30
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(survival, nbins)
    ax.plot(days, dist_exp * len(survival) * smax / nbins, '-r', lw=3)
    ax.set_xlabel("Survival time (days)")
    ax.set_ylabel("Number of patients")
    
    plt.savefig(args.output_exp)
    
# function to fit the Birnbaum-Sanders distribution
def fit_bs(data, args):
    survival = data.survival
    dist = st.fatiguelife
    arguments = dist.fit(survival)
    smax = survival.max()
    days = np.linspace(0., smax, 1000)
    smean = survival.mean()
    rate = 1. / smean
    dist_exp = st.expon.pdf(days, scale=1. / rate)
    
    dist_fl = dist.pdf(days, *arguments)
    nbins = 30
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(survival, nbins)
    ax.plot(days, dist_exp * len(survival) * smax / nbins,
            '-r', lw=3, label='exp')
    ax.plot(days, dist_fl * len(survival) * smax / nbins,
            '--g', lw=3, label='BS')
    ax.set_xlabel("Survival time (days)")
    ax.set_ylabel("Number of patients")
    ax.legend()
    print(args.output_bs)
    
    plt.savefig(args.output_bs)
    
def main():
    args = parser_arguments()
    datafile = args.csv
    data = pd.read_csv(datafile)
    data = data[data.censors == 1]
    
    if args.plot:
        plot_survival(data, args)
        
    if args.plot_exp:
        fit_exp(data, args)
        
    if args.plot_bs:
        fit_bs(data, args)

if __name__ == '__main__':
    main()