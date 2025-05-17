import fnmatch
import random
import re
import networkx as nx
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
import json
import os
import argparse
from duration_sim import DurationSimulator


parser = argparse.ArgumentParser(description="Run task assignment simulation using DeepWalk and KGE embeddings.")

parser.add_argument('--testing_date', type=str, default='2021-08-17', help='Date to start the task assignment simulation')
parser.add_argument('--original_tables_file', type=str, default='orig_original_table.xlsx', help='Path to original RMA task data')
parser.add_argument('--model_dir', type=str, default='embedding_example', help='Directory to load embeddings and save results')
parser.add_argument('--structure', type=str, default='both', choices=['local', 'global', 'both'],
                    help='Embedding strategy: local (DeepWalk), global (KGE), or both (hybrid)')
parser.add_argument('--lambda_weight', type=float, default=0.5,
                    help='Weight for KGE when using both embeddings (structure=both)')

parser.add_argument('--setting_dir', type=str, default='setting', help='Directory containing task_set.json')
parser.add_argument('--log_directory_path', type=str, default='DATA', help='Path to folder containing .log files')
parser.add_argument('--tmp_dir', type=str, default='models', help='Temporary directory for intermediate files')
parser.add_argument('--theta1', type=float, default=1, help='Threshold for task assignment')
parser.add_argument('--output_dir', type=str, default='results_xlsx', help='Directory to save results')
args = parser.parse_args()

if args.structure == 'local':
    KGE_Weighting = 0
    DeepWalk_Weighting = 1
elif args.structure == 'global':
    KGE_Weighting = 1
    DeepWalk_Weighting = 0
elif args.structure == 'both':
    KGE_Weighting = args.lambda_weight
    DeepWalk_Weighting = 1
else:
    raise ValueError(f"Unknown structure: {args.structure}")


tmp_dir = args.tmp_dir
setting_dir = args.setting_dir
model_dir = args.model_dir
testing_date = args.testing_date

log_directory_path = args.log_directory_path
theta1 = args.theta1
original_tables_file = args.original_tables_file

os.makedirs(args.output_dir, exist_ok=True)
maintain_tables_file = f"{args.output_dir}/deco_{args.structure}_λ{KGE_Weighting}.xlsx"
print(f"maintain_tables_file: {maintain_tables_file}")
# Load task config
task_set = f'{setting_dir}/task_set.json' 
with open(task_set, 'r') as json_file:
    config = json.load(json_file)

workload_limit = config['workload_limit']
all_types = config['all_types']

# Embedding file paths
model_output = 'orig_word2vec'
embedding_output = 'orig_node_embeddings'

G2_output_file = os.path.join(model_dir, 'G2.txt') 

# KG embedding settings
VER = 'DECO_KGE'
EMB_DIM = 128
MARGIN = 1.0
BATCH = 32
EPOCH = 100
LEARNING_RATE = 0.001

KG_FILE = os.path.join(model_dir, 'G2.txt')
NODE_FILE = os.path.join(model_dir, f'{VER}_entity_embeddings.csv')
RELATION_FILE = os.path.join(model_dir, f'{VER}_relation_embeddings.csv')
MODEL_FILE = os.path.join(model_dir, f'{VER}.pth')
DEEPWALK_FILE = os.path.join(model_dir, 'orig_node_embeddings.txt')

def combine_embeddings(deepwalk_embeddings, kge_embeddings):
    """
    combine deepwalk and kge embeddings
    """
    combined_embeddings = {}

    for key in deepwalk_embeddings.key_to_index:
        deepwalk_vec = deepwalk_embeddings.get_vector(key)

        if key in kge_embeddings:
            kge_vec = kge_embeddings[key]
            
            combined_embeddings[key] = DeepWalk_Weighting * deepwalk_vec + KGE_Weighting * kge_vec
        else:
            combined_embeddings[key] = deepwalk_vec

    return combined_embeddings

# ---- Load Embeddings ----
def load_deepwalk_embeddings(embedding_file):
    return KeyedVectors.load_word2vec_format(embedding_file, binary=False)

def load_kge_embeddings(embedding_file):
    
    df = pd.read_csv(embedding_file, header=None, names=['entity', 'embedding'])
    embeddings = {row['entity']: np.array([float(x) for x in row['embedding'].split(',')]) for _, row in df.iterrows()}
    return embeddings


    
def maintain_tables(df, specified_date):
    """
    simulate table status before the assignment period
    """
    # Load the Excel file
    date = str(specified_date)
    # Convert 'CQC Start Date' to datetime format for comparison
    df['CQC Start Date'] = pd.to_datetime(df['CQC Start Date'])

    # Filter the dataframe for records with 'CQC Start Date' before the specified date
    df_before_specified_date = df[df['CQC Start Date'] < pd.to_datetime(date)]

    # Save the filtered data to a new Excel file
    output_file_path = maintain_tables_file
    df_before_specified_date.to_excel(output_file_path, index=False)
    
    return output_file_path

def datesplit(df):
    """
    use CQC_id to split the data by date
    """
    grouped_data = {}
    split_index = int(len(df)*0.75)
    df = df[split_index:]
    df.loc[:, 'CQC Start Date'] = pd.to_datetime(df['CQC Start Date']).dt.date

    # Group data by 'CQC Start Date'
    for date, date_group in df.groupby('CQC Start Date'):
        cqc_ids = date_group['CQC_id'].tolist()
        formatted_date = date.strftime('%Y-%m-%d')
        grouped_data[formatted_date] = cqc_ids

    # save to json
    output_json_path = f'{tmp_dir}/grouped_by_date.json'

    with open(output_json_path, 'w') as json_file:
        json.dump(grouped_data, json_file, indent=4)
        
    print(f'Grouped data has been saved to {output_json_path}')


## ------------------------CALCULATE-----------------------##

def testcase_each_task():
    # access the grouped data
    with open(f'{tmp_dir}/grouped_by_date.json', 'r') as file:
        date_groups = json.load(file)

    # read CQC_id and #failmech
    df_failmech = pd.read_excel(original_tables_file)
    failmech_dict = df_failmech.set_index('CQC_id')['#Fail Mech'].to_dict()

    # initialize a dict to store the log and fail testname of each failmech
    failmech_log_failures = {}
    
    # scan all dates
    for date, cqc_ids in date_groups.items():
        for cqc_id in cqc_ids:
            # search the corresponding log
            for filename in os.listdir(log_directory_path):
                if fnmatch.fnmatch(filename, f"{cqc_id}*.log"):
                    # path combine
                    log_file_path = os.path.join(log_directory_path, filename)
                    
                    # load log
                    try:
                        with open(log_file_path, 'r') as log_file:
                            log_content = log_file.read()
                            test_status_pattern = r'\d+\s+\d+\s+(.*?)\s+.*?\s+(passed|FAILED)'
                            tests = set(re.findall(test_status_pattern, log_content))
                            failed_test_names = [test[0] for test in tests if test[1] == 'FAILED' and len(test[0]) >= 5]
                        
                            # get failmech from dict
                            failmech = failmech_dict.get(cqc_id)
                            if failmech:
                                if failmech not in failmech_log_failures:
                                    failmech_log_failures[failmech] = {}
                                failmech_log_failures[failmech][filename] = failed_test_names
                                
                    except FileNotFoundError:
                        continue

    # save failmech log failures to json
    fail_json_path = f'{tmp_dir}/Fail_mech_log_failures.json'
    with open(fail_json_path, 'w') as json_file:
        json.dump(failmech_log_failures, json_file, indent=4)
    
    print(f'Data has been saved to {fail_json_path}')

## ------------------------TRAINING-----------------------##
def add_nodes_edges(G, data, pass_fail_label):
    """
    add nodes and edges to the graph
    """
    for type_key, tasks in data.items():
        if type_key not in G:
            G.add_node(type_key, layer='Type')  
        for task_name, testcases in tasks.items():
            task_node = f"{task_name}"
            if task_node not in G:
                G.add_node(task_node, layer='Task', type=type_key)  
                G.add_edge(type_key, task_node)
            for testcase in testcases:
                testcase_node = f"{testcase}"
                if (task_node,testcase_node) not in G.edges:
                    if testcase_node not in G:
                        G.add_node(testcase_node, layer=f"{pass_fail_label}")
                    G.add_edge(task_node,testcase_node)

def initgraph():
    """
    build graph ; 分pass/fail_{testname}
    """
    G = nx.DiGraph()
    with open(f'{tmp_dir}/Fail_mech_log_failures.json', 'r') as file:
        fail_data = json.load(file)
    add_nodes_edges(G,fail_data, 'FAILED')
    return G

## ------------------------WORKLOAD-----------------------##
def calculate_workload(file_path, specific_date):
    """
    calculate the workload of each user
    """
    df = pd.read_excel(file_path)
    # Calculate 'CQC Close Date'
    df['CQC Close Date'] = df['CQC Start Date'] + pd.to_timedelta(df['Duration (Days)'], unit='D')
    
    
    columns = pd.unique(df['CQE'])
    # Initialize an empty DataFrame to store daily workloads for each user
    workloads = pd.DataFrame(0, index=pd.date_range(start=df['CQC Start Date'].min(), end=df['CQC Close Date'].max()), 
                            columns=columns).astype(float)

    # Iterate through the original DataFrame to update the workloads DataFrame
    for _, row in df.iterrows():
        active_dates = pd.date_range(start=row['CQC Start Date'], end=row['CQC Close Date'])
        workloads.loc[active_dates, row['CQE']] += 1
    # Convert specific_date to pd.Timestamp to ensure compatibility
    specific_date = pd.to_datetime(specific_date)
    
    # Check if the specific date is within the DataFrame index (date range)
    if specific_date in workloads.index:
        # Retrieve the workload for the specific date and convert it to a dictionary
        workload_for_date = workloads.loc[specific_date].to_dict()
    else:
        workload_for_date = {}
        print(f"No workload data available for {specific_date}.")
    return workload_for_date

## ------------------------SIMIRAITY-----------------------##
def fail_similarity(cqc_date, G, combined_embeddings):
    # Load the CQC data for the specific date
    with open(f'{tmp_dir}/grouped_by_date.json', 'r') as file:
        cqc_data = json.load(file)
    
    cqc_list = []
    cqc_date = str(cqc_date)
    total_failmech_result = []
    total_failmech_score = []
    
    # Check if the specific date exists in the data
    if cqc_date in cqc_data:
        cqc_ids = cqc_data[cqc_date]

        for cqc_id in cqc_ids:
            cqc_list.append(cqc_id)
            test_cases = extract_test_cases(cqc_id)  

            if test_cases:
                combined_scores = get_similarity_scores(cqc_id, test_cases, G, combined_embeddings)
            else:
                combined_scores = {type_name: random.random() for type_name in all_types}

            # Normalize scores
            min_score = min(combined_scores.values())
            max_score = max(combined_scores.values())
            normalized_scores = {type_name: (score - min_score) / (max_score - min_score) for type_name, score in combined_scores.items()}
            
            # Convert normalized scores to probabilities
            total_score = sum(normalized_scores.values())
            probability_scores = {type_name: score / total_score for type_name, score in normalized_scores.items()}
            
            # Sort the results
            sorted_items = sorted(probability_scores.items())
            sorted_types = [item[0] for item in sorted_items]
            sorted_scores = [item[1] for item in sorted_items]

            # Collect all sorted scores
            total_failmech_result.append(sorted_types)
            total_failmech_score.append(sorted_scores)

    return cqc_list, total_failmech_result, total_failmech_score


def extract_test_cases(cqc_id):
    """
    extract the testname of cqc_id
    """
    test_cases = []
    test_status_pattern = r'\d+\s+\d+\s+(.*?)\s+.*?\s+(passed|FAILED)'
    # use fnmatch to find the log that matches cqc_id
    for filename in os.listdir(log_directory_path):
        if fnmatch.fnmatch(filename, f"{cqc_id}*.log"):
            log_file_path = os.path.join(log_directory_path, filename)
            
            # read the log and search the test status
            try:
                with open(log_file_path, 'r') as log_file:
                    log_content = log_file.read()
                    # use regex to find all matches
                    matches = re.findall(test_status_pattern, log_content)
                    # add the testname and status to the list
                    for match in matches:
                        test_name, status = match
                        test_cases.append(test_name)
                        
            except FileNotFoundError:
                print(f"File not found: {log_file_path}")
            except Exception as e:
                print(f"Error reading {log_file_path}: {e}")

    return test_cases

def get_similarity_scores(cqc_id, test_cases, G, combined_embeddings):
    scores = {}
    
    # Collect type vectors from the combined embeddings
    type_vectors = {type_key: combined_embeddings[type_key] for type_key in G.nodes if type_key in combined_embeddings and G.nodes[type_key]['layer'] == 'Type'}
    
    # Collect test vectors from the combined embeddings
    test_vectors = [combined_embeddings[case] for case in test_cases if case in combined_embeddings]

    if not test_vectors or not type_vectors:
        print("No valid test vectors or type vectors available.")
        return {}

    # Convert to matrices
    test_matrix = np.array(test_vectors, ndmin=2)
    
    type_names = list(type_vectors.keys())
    type_matrix = np.array([type_vectors[name] for name in type_names], ndmin=2)

    # Compute cosine similarity
    if test_matrix.size == 0 or type_matrix.size == 0:
        print("Empty test matrix or type matrix.")
        return {}

    similarities = cosine_similarity(test_matrix, type_matrix)
    # Calculate scores
    for i, type_name in enumerate(type_names):
        scores[type_name] = np.sum(similarities[:, i])

    return scores


## ------------------------TP table-----------------------##
def calculate_fail_mech_counts(file_path,cqc_date):
    """calculate the number of times each user handles each fail mech before cqc_date."""
    df = pd.read_excel(file_path)
    cqc_datetime = pd.to_datetime(cqc_date) 
    # filter the data before cqc_datetime
    df_filtered = df[df['CQC Start Date'] <= cqc_datetime] 
    # group by CQE and #Fail Mech and count
    fail_mech_counts = df_filtered.groupby(['CQE', '#Fail Mech']).size().unstack(fill_value=0).to_dict(orient='index')
    return fail_mech_counts

def calculate_user_speed(file_path,cqc_date):
    """calculate the average processing time of each user for each fail mech before cqc_date."""
    # filter the data before cqc_datetime
    df = pd.read_excel(file_path)
    cqc_datetime = pd.to_datetime(cqc_date) 
    df_filtered = df[df['CQC Start Date'] <= pd.to_datetime(cqc_datetime)]
    
    # calculate the maximum processing days of each fail mech (among all users)
    max_duration_by_mech = df_filtered.groupby('#Fail Mech')['Duration (Days)'].max()
    
    # group by CQE and #Fail Mech and calculate the average processing time
    avg_duration_by_mech = df_filtered.groupby(['CQE', '#Fail Mech'])['Duration (Days)'].mean().unstack(fill_value=0)
    
    user_speed = {}
    for user, mech_durations in avg_duration_by_mech.iterrows():
        user_speed[user] = {
                    mech: 1 - (duration / max_duration_by_mech[mech]) if max_duration_by_mech[mech] > 0 and duration > 0 else 0 
                    for mech, duration in mech_durations.items()
                }
    return user_speed

def calculate_workload_ratio(current_workload):
    """calculate the workload rate based on current workload and limit"""
    workload_ratio = {}
    for user, limit in workload_limit.items():
        workload = current_workload.get(user, 0)
        ratio = workload / limit if (limit > 0 and workload>0) else 0.01
        workload_ratio[user] = ratio
    return workload_ratio

def calculate_S_fitness(maintain_table, cqc_date):
    """calculate the S_fitness score"""

    fail_mech_counts = calculate_fail_mech_counts(maintain_table, cqc_date)
    user_speed = calculate_user_speed(maintain_table, cqc_date)
    df_fail_mech_counts = pd.DataFrame.from_dict(fail_mech_counts, orient='index').fillna(0).astype(int)
    df_relative_speed = pd.DataFrame.from_dict(user_speed, orient='index').fillna(0)
    tester_vector = pd.DataFrame(index=df_fail_mech_counts.index, columns=df_fail_mech_counts.columns)
    
    for user in df_fail_mech_counts.index:
        for fail_mech in df_fail_mech_counts.columns:
            N_type_norm = df_fail_mech_counts.at[user, fail_mech] / df_fail_mech_counts.loc[user].max() 
            T_avg_type_norm = df_relative_speed.at[user, fail_mech] 

            S_fitness = N_type_norm + T_avg_type_norm 
            tester_vector.at[user, fail_mech] = S_fitness
           

    # Normalize the scores so that the sum for each user is 1
    for user in tester_vector.index:
        min_score = tester_vector.loc[user].min()
        max_score = tester_vector.loc[user].max()
        if max_score != min_score:
            tester_vector.loc[user] = (tester_vector.loc[user] - min_score) / (max_score - min_score)
        else:
            tester_vector.loc[user] = 1 / len(tester_vector.columns)
    # Normalize the scores so that the sum for each user is 1
    for user in tester_vector.index:
        user_scores = tester_vector.loc[user]
        sum_scores = user_scores.sum()
        if sum_scores != 0:  # Avoid division by zero
            normalized_scores = user_scores / sum_scores
            tester_vector.loc[user] = normalized_scores
        else:
            tester_vector.loc[user] = 1 / len(user_scores)  # If all scores are zero, distribute evenly
    
    # Transpose the DataFrame to switch rows and columns
    tester_vector = tester_vector.transpose()
    
    return tester_vector

def calculate_user_task_score(user, task_vector, S_fitness_scores, fail_mech_counts):
    """calculate the user's task score based on the new calculation logic"""
    df_fail_mech_counts = pd.DataFrame.from_dict(fail_mech_counts, orient='index').fillna(0).astype(int)
    count = 0
    X_total = 0

    for fail_mech in task_vector.index:
        N_type_norm = df_fail_mech_counts.at[user, fail_mech] 
        if N_type_norm:
            count += 1
            user_vector = S_fitness_scores.at[fail_mech, user]
            
            task_value = task_vector.at[fail_mech]
            X_k = np.log10(user_vector+1) - np.log10(task_value+1) 
            X_total += X_k
            
    if count > 0:
        score = X_total / count
    else:
        score = float(0)
    return score

def calculate_adaptability_scores(maintain_table, cqc_date, current_workload, type_list, type_prob, cqc_list):
    """based on S_fitness scores, calculate the adaptability scores"""
    S_fitness_scores = calculate_S_fitness(maintain_table, cqc_date)
    
    adaptability_scores_df = pd.DataFrame(0.0, index=S_fitness_scores.columns, columns=cqc_list)
    
    # a dict to store all task_vector, key is cqc ID
    all_task_vectors = {}
    all_tester_vector = {}
    
    # use for loop to ensure each cqc is processed
    for i, cqc in enumerate(cqc_list):
        # if type_list or type_prob list is not long enough, set it to empty list
        fail_mechs = type_list[i] if i < len(type_list) else []
        probs = type_prob[i] if i < len(type_prob) else []
        
        S1 = {}  # each cqc has a separate S1 set
        
        # combine fail_mechs and probs to form a task vector
        task_vector = pd.Series(probs, index=fail_mechs, dtype=np.float64)
        
        # convert task_vector to a dict and store it in all_task_vectors
        task_vector_dict = {str(k): float(v) for k, v in task_vector.to_dict().items()}
        all_task_vectors[cqc] = task_vector_dict
        
        # convert S_fitness_scores to a dict and store it in all_tester_vector (ensure it can be serialized by JSON)
        all_tester_vector[cqc] = S_fitness_scores.to_dict()
        
        # iterate over each user
        for user in S_fitness_scores.columns:
            # combine the scores of the user's fail_mechs to form a user vector
            user_vector = S_fitness_scores.loc[fail_mechs, user].to_numpy(dtype=np.float64)
            # calculate the adaptability score using JSD
            jsd = jensenshannon(task_vector, user_vector, base=2)
            # update the score in the DataFrame
            adaptability_scores_df.at[user, cqc] += float(jsd)
            
            # if jsd is less than the preset threshold theta1, add the user to S1 set
            if float(jsd) < theta1:
                if user not in S1:
                    S1[user] = []
                S1[user].append((cqc, float(jsd)))
        
        # recalculate the user's task score in S1
        fail_mech_counts = calculate_fail_mech_counts(maintain_table, cqc_date)
        for user in S1:
            # here, based on the current cqc index, rebuild the task vector
            task_vector = pd.Series(type_prob[i], index=type_list[i], dtype=np.float64)
            new_score = calculate_user_task_score(user, task_vector, S_fitness_scores, fail_mech_counts)
            adaptability_scores_df.at[user, cqc] = new_score
        
        # set the score of the users not in S1 to negative infinity
        for user in adaptability_scores_df.index:
            if user not in S1:
                adaptability_scores_df.at[user, cqc] = float('-inf')
    
    # if you want to accumulate the results of multiple calls, try to read the existing JSON file first
    if os.path.exists(os.path.join(model_dir, "all_task_vectors.json")):
        with open(os.path.join(model_dir, "all_task_vectors.json"), "r") as json_file:
            existing_vectors = json.load(json_file)
    else:
        existing_vectors = {}

    # merge the all_task_vectors produced this time
    existing_vectors.update(all_task_vectors)
    
    # save the merged results to a JSON file
    with open(os.path.join(model_dir, "all_task_vectors.json"), "w") as json_file:
        json.dump(existing_vectors, json_file, indent=4)
    
############################################### tester vector
    if os.path.exists(os.path.join(model_dir, "all_tester_vector.json")) and os.path.getsize(os.path.join(model_dir, "all_tester_vector.json")) > 0:
        try:
            with open(os.path.join(model_dir, "all_tester_vector.json"), "r") as json_file:
                existing_tester_vectors = json.load(json_file)
        except json.JSONDecodeError:
            print("the content of all_tester_vector.json is not correct, initialize it to an empty dictionary.")
            existing_tester_vectors = {}
    else:
        existing_tester_vectors = {}

    # merge the all_tester_vector produced this time
    existing_tester_vectors.update(all_tester_vector)

    # save the merged results to a JSON file
    with open(os.path.join(model_dir, "all_tester_vector.json"), "w") as json_file:
        json.dump(existing_tester_vectors, json_file, indent=4)
    
    output_data = {str(cqc_date): adaptability_scores_df.to_dict()}
    return adaptability_scores_df

cqc_score_record = {}  
## ------------------------Assignment-----------------------##
def assign_tasks_based_on_urgency_and_workload(TP_table,cqc_list,current_workload,cqc_date,type_list,maintain_table):
    '''perform assignment'''
    adaptability_scores_df = TP_table
    threshold = -1
    task_assignments = pd.DataFrame(index=adaptability_scores_df.index, columns=cqc_list).fillna('')
    cqc_urgent_pairs = list(cqc_list)
    workload_ratio = calculate_workload_ratio(current_workload)
    assigned_users = []

    i=0
    for cqc in cqc_urgent_pairs:
        if cqc in adaptability_scores_df.columns:  # Check if exist

            # filter out users with workload_ratio >= 1
            eligible_users = adaptability_scores_df.index.tolist()
            # eligible_users = [user for user in adaptability_scores_df.index if workload_ratio.get(user, 0) < 1]
            cqc_scores = adaptability_scores_df.loc[eligible_users, cqc].copy()
            for user in cqc_scores.index:
                # Adjust the adaptability score considering the workload ratio, for eligible users only

                # if workload_ratio < 0.5, reverse cqc_scores
                if workload_ratio.get(user, 0) < threshold:
                    if random.random() < 0.2 and cqc_scores[user]>0:
                        cqc_scores[user] = 1 - cqc_scores[user]
                        # print(f"user: {user}, newscore: {cqc_scores[user] }")

            # find the user with the highest TP and zero workload
            zero_workload_users = [user for user in cqc_scores.index if current_workload.get(user, 0) == 0 and user not in assigned_users]
            best_user = None
            if zero_workload_users:
                zero_workload_scores = cqc_scores[zero_workload_users]
                
                if not zero_workload_scores.empty:
                    # TODO save the highest score to cqc : score
                    best_user = zero_workload_scores.idxmax()
                    print(f"zero_workload_scores: {zero_workload_scores.max()}")
                    cqc_score_record[cqc] = {"score": zero_workload_scores.max()}
                    assigned_users.append(best_user)

            if best_user is None and not cqc_scores.empty:
                best_user = cqc_scores.idxmax()
                cqc_score_record[cqc] = {"score": cqc_scores.max()}
                print(f"cqc_scores: {cqc_scores.max()}")

            print(f"CQC: {cqc}, Best User: {best_user}, Highest Score: {cqc_score_record[cqc]}")
            if best_user:
                task_assignments.at[best_user, cqc] = 'Assigned'
                update_table(maintain_table, cqc, best_user, cqc_date)
                current_workload[best_user] = current_workload.get(best_user, 0) + 1  
                
            else:
                print(f"No suitable user found for task '{cqc}' due to workload constraints.")
        else:
            print(f"Warning: '{cqc}' not found in adaptability_scores_df columns")
        i += 1
    return task_assignments

def update_table(path, cqc_id, user, start_date):
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"Error loading the Excel file: {e}")
        return
    simulator = DurationSimulator(workload_limit)
    real_failmech, duration = simulator.simulate_duration_by_id(df, cqc_id, user, start_date, original_tables_file)
    start_date_formatted = pd.to_datetime(start_date)
    close_date = start_date_formatted + pd.to_timedelta(duration, unit='D')
    new_record = {'CQC_id': [cqc_id], 
                'CQE': [user], 
                '#Fail Mech':[real_failmech],
                'CQC Start Date': [start_date_formatted],
                'Duration (Days)':[duration],
                'CQC Close Date':[close_date]
                }
    new_record_df = pd.DataFrame(new_record)

    updated_df = pd.concat([df, new_record_df], ignore_index=True)
    
    try:
        updated_df.to_excel(path, index=False)
        print("Updated and saved successfully.")
    except Exception as e:
        print(f"Error saving the updated DataFrame: {e}")

def main():
    ##### initial setting 
    path = original_tables_file
    df = pd.read_excel(path)
    df['CQC Start Date'] = pd.to_datetime(df['CQC Start Date'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_dir = 'results_xlsx'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #### set started time
    specified_date = pd.Timestamp(testing_date)
    df_filtered_sorted = df[df['CQC Start Date'] >= specified_date].sort_values(by='CQC Start Date')
    unique_dates = df_filtered_sorted['CQC Start Date'].dt.date.unique()
    
    #### start from the last 25% of the tasks
    maintain_table = maintain_tables(df, specified_date)
    
    #### CQC are divided by date ####
    datesplit(df)
    
    #### calculate the pass/fail testcase in each cqc ####
    testcase_each_task()
    
    ####### TRAINING (DeepWalk) #######
    print("----------Training DeepWalk---------\n")
    Graph = initgraph()################

    ####### Combine Embeddings #######
    print("----------Combine DeepWalk and KGE Embeddings---------\n")
    deepwalk_embeddings = load_deepwalk_embeddings(DEEPWALK_FILE)
    kge_embeddings = load_kge_embeddings(NODE_FILE)
    combined_embeddings = combine_embeddings(deepwalk_embeddings, kge_embeddings)


    # every date starts with cqc assign
    for cqc_date in unique_dates:
        print("\n--------------\nDate: ")
        print(cqc_date)
        
        print("----------Workload---------\n")
        current_workload = calculate_workload(maintain_table, cqc_date)
        
        print("----------Similarity---------\n")
        cqc_list, type_list, type_prob = fail_similarity(cqc_date, Graph, combined_embeddings)
    
        print("----------TP table--------\n")
        TP_table = calculate_adaptability_scores(maintain_table, cqc_date, current_workload, type_list, type_prob, cqc_list)
        
        ##### Assignment
        print("----------Assignment---------\n")
        assign_tasks_based_on_urgency_and_workload(TP_table, cqc_list, current_workload, cqc_date, type_list, maintain_table)
    with open(os.path.join(model_dir, "cqc_score_record.json"), "w") as json_file:
        json.dump(cqc_score_record, json_file, indent=4)
    print("CQC Score Record has been saved to cqc_score_record.json")

if __name__ == "__main__":
    main()
    # read the JSON file
    with open(os.path.join(model_dir, "all_task_vectors.json"), "r") as json_file:
        all_task_vectors = json.load(json_file)
    # count the number of cqc (the number of keys in the dictionary)
    num_cqc1 = len(all_task_vectors)
    print(f"the number of cqc in all_task_vectors.json: {num_cqc1}")
    with open(os.path.join(model_dir,"all_tester_vector.json"), "r") as json_file:
        all_tester_vector = json.load(json_file)

    # count the number of cqc (the number of keys in the dictionary)
    num_cqc2 = len(all_tester_vector)
    print(f"the number of cqc in all_tester_vector.json: {num_cqc2}")