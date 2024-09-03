import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


# Function to test the model and visualize the outputs individually
def test_and_visualize(model, test_loader, feature_names):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, y = batch
            y_hat = model(x)
            inputs = x.numpy()
            outputs = y_hat.numpy()
            errors = (y_hat - y).numpy()
  
            for j in range(inputs.shape[0]):
                input_sample = inputs[j]
                output_sample = outputs[j]
                error_sample = errors[j]
                
                plt.figure(figsize=(12, 6))

                # Plot input vs output
                plt.subplot(1, 2, 1)
                indices = np.arange(len(input_sample))
                plt.bar(indices - 0.2, input_sample, width=0.4, label='Input')
                plt.bar(indices + 0.2, output_sample, width=0.4, label='Output', alpha=0.7)
                plt.xticks(indices, feature_names, rotation='vertical')  # Set X ticks to feature names
                plt.title(f'Sample {i*test_loader.batch_size + j} - Input vs Output')
                plt.legend()

                # Plot error
                plt.subplot(1, 2, 2)
                plt.bar(indices, error_sample)
                plt.xticks(indices, feature_names, rotation='vertical')  # Set X ticks to feature names
                plt.title(f'Sample {i*test_loader.batch_size + j} - Error')

                plt.show()


def compare_conditions(df1, df2, df_label1='DF1', df_label2='DF2', title='Absolute Error by Sample ID from Two DataFrames'):
    df1['sample_id'] = df1['sample_id'].astype(str)
    df2['sample_id'] = df2['sample_id'].astype(str)
    df1['ground_truth'] = df1['ground_truth'].astype(float)
    df1['predictions'] = df1['predictions'].astype(float)  # Ensure prediction is float
    
    merged_df = pd.merge(df1, df2, on='sample_id', suffixes=('_df1', '_df2'))
    
    mean_ground_truth = merged_df['ground_truth_df1'].mean()
    merged_df['ground_truth_deviation'] = merged_df['ground_truth_df1'] - mean_ground_truth
    merged_df['predictions_deviation'] = merged_df['predictions_df1'] - mean_ground_truth  # Calculate prediction deviation
    
    # Increase the figure's width for better label spacing
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    width = 0.35
    ind = np.arange(len(merged_df))
    
    # Plotting Absolute Error
    axes[0].bar(ind, merged_df['abs_error_df1'], width, label=df_label1)
    axes[0].bar(ind + width, merged_df['abs_error_df2'], width, label=df_label2)
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title(title)
    axes[0].set_xticks(ind + width / 2)
    axes[0].set_xticklabels(merged_df['sample_id'], rotation=45, ha='right')
    axes[0].legend()
    
    # Plotting Ground Truth and Prediction Deviation
    axes[1].bar(ind - width/2, merged_df['ground_truth_deviation'], width, color='lightgreen', label='Ground Truth Deviation from Mean')
    axes[1].bar(ind + width/2, merged_df['predictions_deviation'], width, color='lightblue', label='Prediction Deviation from Mean')
    axes[1].set_xlabel('Sample ID')
    axes[1].set_ylabel('Deviation from Mean Ground Truth')
    axes[1].set_title('Deviation from Mean Ground Truth and Prediction')
    axes[1].set_xticks(ind)
    axes[1].set_xticklabels(merged_df['sample_id'], rotation=45, ha='right')
    axes[1].axhline(y=0, color='gray', linewidth=1, linestyle='--')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()



def plot_mean_abs_error(df_dict):
    """
    Plots the mean abs_error of each DataFrame in the dictionary.

    Parameters:
    - df_dict: A dictionary of DataFrames with 'abs_error' columns.

    The function calculates the mean 'abs_error' of each DataFrame and
    plots these means in a bar chart, using the dictionary keys as bar labels.
    """
    
    # Calculate the mean abs_error for each DataFrame and store it in a new dictionary
    means = {key: df['abs_error'].mean() for key, df in df_dict.items()}
    
    # Set up the plotting figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for plotting
    labels = list(means.keys())
    mean_values = list(means.values())
    
    # Create a list of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # Create bars with different colors
    ax.bar(labels, mean_values, color=colors)
    
    # Adding some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('DataFrame')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Mean Absolute Error by DataFrame')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    
    # Show plot
    plt.xticks(rotation=45)  # Rotate labels to make them readable
    plt.tight_layout()
    plt.show()


