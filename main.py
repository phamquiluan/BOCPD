from __future__ import division
import matplotlib.pyplot as plt
import bayesian_changepoint_detection.generate_data as gd
import seaborn
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.cm as cm
import numpy as np
import json
import os
import glob
from os.path import basename,join,dirname
from datetime import datetime
import numpy as np
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.offline_likelihoods import IndepentFeaturesLikelihood
import bayesian_changepoint_detection.online_likelihoods as online_ll
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection 
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from functools import partial
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.priors import const_prior
from functools import partial

from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
# ifgnore all warning
import warnings
warnings.filterwarnings("ignore")


def drop_constant(df: pd.DataFrame):
    return df.loc[:, (df != df.iloc[0]).any()]


def read_data(data_path = None):
    data = pd.read_csv(data_path)
    selected_cols = []
    for c in data.columns:
        if 'queue-master' in c or 'rabbitmq_' in c: continue 
        if "latency-50" in c or "_error" in c:
            selected_cols.append(c)

    data = data[selected_cols]
    data = drop_constant(data)
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    for c in data.columns:
        data[c] = (data[c] - np.min(data[c])) / (np.max(data[c]) - np.min(data[c]))
    data = data.fillna(method="ffill")
    data = data.fillna(0)
    return data


def bocpd(data : pd.DataFrame):
    anomalies = []
    for col in data.columns:
        values = data[col]

        prior_function = partial(const_prior, p=1/(len(values) + 1))
        Q, P, Pcp = offline_changepoint_detection(values, prior_function ,offline_ll.StudentT())
        anomalies.extend(np.where(np.exp(Pcp).sum(0) > 0.6)[0].tolist())
    # merge continuous timepoint
    anomalies = sorted(anomalies)
    merged_anomalies = [anomalies[i] for i in range(len(anomalies)) if i == 0 or anomalies[i] - anomalies[i-1] > 1]
    return merged_anomalies


def worker(data_path):
    service_metric = basename(dirname(dirname(data_path)))
    case_idx = basename(dirname(data_path))
    data_dir = dirname(data_path)
    
    # PREPARE DATA
    data = read_data(data_path)   
    normal_data = data.iloc[30:330]
    output = bocpd(normal_data)

    with open(f"./output/{service_metric}_{case_idx}.json", "w") as f:
        json.dump(output, f)

from multiprocessing.pool import Pool

with Pool(2) as p:
    list(tqdm(p.imap(worker, glob.glob("../cfm/data/fse-ob/**/simple_data.csv", recursive=True))))
