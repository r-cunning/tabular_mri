



















def plot_avg_experiment_error(self, x_label='Condition', y_label='Metric', title='Metric Comparison per Condition'):
    from src.visualization.plots import plots
    plots.plot_dict_values(self.run_errors_mean, x_label=x_label, y_label=y_label, title=title, std_dict=self.run_errors_std)
    self.plots['run_errors'] = plt.gcf()
    
# def plot_single_sample_errors(self, sample_id, target_column='abs_error'):
    
#     experiments = self.sample_errors[sample_id]['run_name']
#     errors = self.sample_errors[sample_id]['abs_error']
    
def plot_experiment_errors_by_sample(self):
    from src.visualization.plots import plots
    plots.plot_experiment_errors(self.reports, 'val_id', 'abs_error')
    self.plots['experiment_errors_by_subject'] = plt.gcf()

def plot_single_sample_errors(self, sample_id, target_column='abs_error', bar_width=None):
    # Check if sample_id is in the data
    if sample_id not in self.sample_errors:
        raise ValueError(f"Sample ID {sample_id} not found in data.")

    # Extract experiments and corresponding errors for the sample
    sample_data = self.sample_errors[sample_id]
    experiments = sample_data['run_name'].unique()
    errors = [sample_data[sample_data['run_name'] == exp][target_column].mean() for exp in experiments]

    # Create a dictionary from experiments to errors
    error_dict = dict(zip(experiments, errors))

    # Plotting
    # bar_width = 0.6  # Adjust bar width here
    plt.figure(figsize=(10, 6))
    
    if bar_width != None:
        for i, (exp, error) in enumerate(error_dict.items()):
            plt.bar(exp, error, bar_width, color=plt.cm.viridis(i / len(error_dict)))
    else:
        for i, (exp, error) in enumerate(error_dict.items()):
            plt.bar(exp, error, color=plt.cm.viridis(i / len(error_dict)))
    
    # Scaling the X axis for neat separation of bars
    plt.xticks(range(len(experiments)), experiments)

    # Adding labels and title with increased font size
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel(f'{target_column.title()} (Average)', fontsize=12)
    plt.title(f'Error for Sample ID {sample_id} Across Experiments', fontsize=14)

    # Showing the plot
    plt.show()

    
    
    
def plot_avg_sample_errors(self, target_column='abs_error'):
    """
    Plot the average sample errors per subject across all experiments.

    Parameters:
    - target_column (str): The name of the column containing the target values. Default is 'abs_error'.

    Raises:
    - ValueError: If the DataFrame for any experiment is missing the required columns.

    Returns:
    - None
    """
    
    # Initialize an empty DataFrame for storing average errors per subject
    avg_error_per_subject = pd.DataFrame()
    std_error_per_subject = pd.DataFrame()
    sub_id_column = 'val_id'
    for exp_name, df in self.reports.items():
        # Ensure the DataFrame has the required columns
        if sub_id_column not in df.columns or target_column not in df.columns:
            raise ValueError(f"DataFrame for experiment '{exp_name}' is missing required columns.")

        # Calculate average error per subject for the experiment
        df[f'abs_error_{exp_name}'] = df[target_column].abs()
        avg_error_exp = df.groupby(sub_id_column)[f'abs_error_{exp_name}'].mean().rename(f'avg_abs_error_{exp_name}')
        # Append to the avg_error_per_subject DataFrame
        if avg_error_per_subject.empty:
            avg_error_per_subject = avg_error_exp
        else:
            avg_error_per_subject = pd.concat([avg_error_per_subject, avg_error_exp], axis=1)

    # Calculate the overall average error per subject across all experiments
    avg_error_per_subject['overall_avg_error'] = avg_error_per_subject.mean(axis=1)
    std_error_per_subject['overall_std_error'] = avg_error_per_subject.std(axis=1)

    # Adjust standard error to not go below zero
    lower_error = std_error_per_subject['overall_std_error'].copy()
    for sub_id in avg_error_per_subject.index:
        lower_error[sub_id] = min(std_error_per_subject.loc[sub_id, 'overall_std_error'], avg_error_per_subject.loc[sub_id, 'overall_avg_error'])

    # Creating a unique color for each subject ID using a colormap
    subjects = avg_error_per_subject.index.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(subjects)))
    color_dict = dict(zip(subjects, colors))

    # Plotting
    bar_width = 0.6  # Adjust bar width here
    for sub_id in subjects:
        plt.bar(sub_id, avg_error_per_subject.loc[sub_id, 'overall_avg_error'], alpha=0.8, width=bar_width, color=color_dict[sub_id])
        # Adding error bars
        plt.errorbar(sub_id, avg_error_per_subject.loc[sub_id, 'overall_avg_error'], yerr=[[lower_error[sub_id]], [std_error_per_subject.loc[sub_id, 'overall_std_error']]], fmt='o', color='black')
        # Annotating with standard deviation
        plt.text(sub_id, avg_error_per_subject.loc[sub_id, 'overall_avg_error'] + std_error_per_subject.loc[sub_id, 'overall_std_error'], f'{std_error_per_subject.loc[sub_id, "overall_std_error"]:.2f}', ha='center')

    # Adjusting the x-axis labels
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align them

    # Adding labels and title
    plt.xlabel('Subject ID')
    plt.ylabel('MAE Across Experiments')
    plt.title('Overall MAE per Subject Across All Experiments')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()