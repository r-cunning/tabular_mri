import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd



class BaseExperiment:
    def __init__(self, TorchRuns: dict):
        self.runs = TorchRuns
        self.run_results = {}
        self.loss_hists = {}
        self.reports = {}
        
        
        
    def run(self):
        for run_name, runs in self.runs.items():
            print("Starting run: ", run_name)
            self.loss_hists[run_name] = {}
            self.run_results[run_name], self.loss_hists[run_name], self.reports[run_name] = runs.train()
            self.reports[run_name] = runs.get_report()