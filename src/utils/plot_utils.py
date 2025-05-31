# src/utils/plot_utils.py

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_regret_curves(results_df: pd.DataFrame, output_dir: str = "data/plots", filename: str = "regret_curves.png"):
    """
    Plots the average regret curves for different algorithms as a function of the budget B.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
                                   Expected columns: 'algorithm', 'budget', 'avg_regret', 'std_regret'.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")
    if not all(col in results_df.columns for col in ['algorithm', 'budget', 'avg_regret', 'std_regret']):
        raise ValueError("results_df must contain 'algorithm', 'budget', 'avg_regret', 'std_regret' columns.")

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 6))

    algorithms = results_df['algorithm'].unique()

    for algo in algorithms:
        algo_data = results_df[results_df['algorithm'] == algo].sort_values(by='budget')
        budgets = algo_data['budget']
        avg_regret = algo_data['avg_regret']
        std_regret = algo_data['std_regret']

        plt.plot(budgets, avg_regret, marker='o', linestyle='-', label=f'{algo} (Avg Regret)')
        plt.fill_between(budgets, avg_regret - std_regret, avg_regret + std_regret, alpha=0.2)

    plt.xscale('log') # Budgets are often log-scaled
    plt.yscale('log') # Regret can also be log-scaled to show O(log B) behavior

    plt.xlabel('Budget (B)')
    plt.ylabel('Average Regret')
    plt.title('Average Regret vs. Budget for Bandit Algorithms')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    plt.savefig(output_filepath)
    print(f"Regret curves plot saved to: {output_filepath}")
    plt.close() # Close the plot to free memory