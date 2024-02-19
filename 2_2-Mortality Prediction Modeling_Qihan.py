###########################################################################################################
#                                       Mortality Modeling
###########################################################################################################
# 
# Licensed under the Apache License, Version 2.0**
# You may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is 
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
# implied. See the License for the specific language governing permissions and limitations under the License.

# -> Authors:
#      Luis R Soenksen (<soenksen@mit.edu>),
#      Yu Ma (<midsumer@mit.edu>),
#      Cynthia Zeng (<czeng12@mit.edu>),
#      Leonard David Jean Boussioux (<leobix@mit.edu>),
#      Kimberly M Villalobos Carballo (<kimvc@mit.edu>),
#      Liangyuan Na (<lyna@mit.edu>),
#      Holly Mika Wiberg (<hwiberg@mit.edu>),
#      Michael Lingzhi Li (<mlli@mit.edu>),
#      Ignacio Fuentes (<ifuentes@mit.edu>),
#      Dimitris J Bertsimas (<dbertsim@mit.edu>),
# -> Last Update: Dec 30th, 2021

import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import sys
import warnings

from prediction_util import run_models, data_fusion, get_data_dict, get_all_dtypes, parallel_run

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Supply the embedding file
    fname = 'data/cxr_ic_fusion.csv'
    df = pd.read_csv(fname, skiprows=[45051, 45052])

    df_death_small48 = df[((df['img_length_of_stay'] < 48) & (df['death_status'] == 1))]
    df_alive_big48 = df[((df['img_length_of_stay'] >= 48) & (df['death_status'] == 0))]
    df_death_big48 = df[((df['img_length_of_stay'] >= 48) & (df['death_status'] == 1))]

    df_death_small48['y'] = 1
    df_alive_big48['y'] = 0
    df_death_big48['y'] = 0
    df = pd.concat([df_death_small48, df_alive_big48, df_death_big48], axis=0)
    df = df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay',
                  'death_status'], axis=1)

    data_type_dict = get_data_dict(df)
    all_types_experiment = get_all_dtypes()

    print('mortality - all_types_experiment', len(all_types_experiment))

    # Number of Cases： 2047
    results = parallel_run(all_types_experiment, data_type_dict, df, 'mortality', start_index=21)
