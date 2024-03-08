import pandas as pd
import torch
import numpy as np

import shap


import torch
import matplotlib.pyplot as plt


class ShapDeepExplainer:
    def __init__(self, dataset):
        # Assuming dataset.feature_names is a list of feature names
        
        self.test_X = {}
        self.test_y = {}
        self.feature_names = dataset.feature_names

        self.importances_df = pd.DataFrame(columns=['sample_id'] + dataset.feature_names)
        self.shap_values = {}
        self.shap_explainers = {}

        self.shap_explanations = {}
        
        self.report_dict = {}
        self.final_report_df = pd.DataFrame()

    
    def evaluate(self, model, train_loader, test_loader, val_ids, hparams, fold):
            model.eval()  # Ensure the model is in evaluation mode
            val_ids_str = '_'.join(str(i) for i in val_ids)
            
            sample_ids = hparams.dataset.sample_ids.iloc[val_ids]
            sample_ids_str = '_'.join(str(i) for i in sample_ids)
            # Use the entire training set to create a background distribution
            background_examples = next(iter(train_loader))[0][:]  # Assuming the first element is the input features
            
            # Create a DeepExplainer to calculate SHAP values
            explainer = shap.DeepExplainer(model, background_examples)
            
            self.shap_explainers[val_ids_str] = explainer
            
            
            
            
            # Use the entire test set to calculate SHAP values
            test_X = next(iter(test_loader))[0]  # Assuming the first element is the input features
            test_y = next(iter(test_loader))[1]  # Assuming the second element is the ground truth labels
            
            
            shap_values = explainer.shap_values(test_X, check_additivity=False)
            self.shap_values[val_ids_str] = shap_values
            
            # Convert SHAP values to a more manageable form if necessary
            shap_values = np.array(shap_values).reshape(test_X.shape)
            
            shap_explanation = shap.Explanation(values=shap_values,
                                                    base_values=explainer.expected_value,
                                                    data=test_X.detach().numpy(),
                                                    feature_names=self.feature_names)

            
            
            
            row_list = []
            # Create a DataFrame to hold all the information
            report_df = pd.DataFrame(columns=[f"{name}" for name in self.feature_names] + 
                                            [f"{name}_SHAP" for name in self.feature_names] +
                                            ['predictions', 'ground_truth', 'error'])
            
            # print("Length of test_X: ", len(test_X))
            
            for i in range(len(test_X)):
                
                
                val_id = val_ids[i]
                sample_id = sample_ids.iloc[i]
                self.test_X[val_id] = test_X[i]
                self.test_y[val_id] = test_y[i]
                
                # Test the model on the current example
                preds = model(test_X[i])
                error = preds - test_y[i]
                
                
                # Construct a single row for the current test example
                row_data = {}
                
                row_data['experiment_name'] = hparams.experiment_name
                row_data['run_name'] = hparams.run_name
                row_data['model_name'] = hparams.base_model.model_name
                row_data['fold'] = fold
                row_data['val_id'] = val_id
                row_data['sample_id'] = sample_id
                # Add the input feature values
                for j, name in enumerate(self.feature_names):
                    row_data[name] = test_X[i][j].item()
                # Add the SHAP values
                for j, name in enumerate(self.feature_names):
                    row_data[f"{name}_SHAP"] = shap_values[i][j]
                # Add the prediction, actual value, and error
                row_data['predictions'] = np.round(preds.item(), 4)
                row_data['ground_truth'] = test_y[i].item()
                row_data['error'] = np.round(error.item(), 4)
                row_data['abs_error'] = np.round(np.abs(error.item()), 4)
                
                row_list.append(row_data)
                # Append the row to the DataFrame
                # report_df = report_df.append(row_data, ignore_index=True)
                


                
            # Concat and store the report DataFrame in the report dictionary using val_ids as the key
            # report_df = pd.concat(report_df, row_data)
            report_df = pd.DataFrame(row_list).sort_values(by='val_id', ascending=True)
            self.report_dict[val_ids_str] = report_df
            
            self.final_report_df = pd.concat(self.report_dict.values(), ignore_index=True).sort_values(by='val_id', ascending=True)
            return self.final_report_df, shap_explanation

