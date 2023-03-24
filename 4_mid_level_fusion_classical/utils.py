
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from itertools import combinations




def get_experts_weights(df):
	experts = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
	experts_kappas = [[], [], [], []]

	for pair in combinations(range(len(experts)), 2):
		labels_expert_0 = df[experts[pair[0]]].values.tolist()
		labels_expert_1 = df[experts[pair[1]]].values.tolist()
		kappa = cohen_kappa_score(labels_expert_0, labels_expert_1)

		experts_kappas[pair[0]].append(kappa)
		experts_kappas[pair[1]].append(kappa)

	experts_kappas = np.array(experts_kappas).mean(axis=1)
	experts_weights = experts_kappas / np.sum(experts_kappas)

	return experts_weights

def compute_label_01(x, experts_weights):
    labels = np.array([x['Expert 1'], x['Expert 2'], x['Expert 3'], x['Expert 4']]) # possible values [1, 2, 3, 4]
    labels_normalized = ((labels - 1) * 2) - 3 # possible values [-3, -1, +1, +3]

    label = (np.sum(labels_normalized * experts_weights)+3) / 6

    return round(label + 1e-10, 9)