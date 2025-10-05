import datetime
import random
import numpy as np

import tqdm
import torch
import pickle
import os
import json
import pandas as pd

import settings
from retriever import Embedder, Retriever, load_list_of_lists

import csv
from openai import OpenAI

# Function to get OpenAI API key from user
def get_openai_api_key():
    """Get OpenAI API key from environment variable or user input"""
    # First try to get from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("Using OpenAI API key from environment variable OPENAI_API_KEY")
        return api_key
    
    # If not found in environment, ask user for input
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        raise ValueError("OpenAI API key cannot be empty")
    return api_key

# Initialize OpenAI client with user-provided API key
openai_api_key = get_openai_api_key()
client = OpenAI(
    api_key=openai_api_key
)

import math
def haversine(p1, p2):
    """
    Calculate the great circle distance between two points on the Earth (specified in decimal degrees).
    (lat1, lon1), (lat2, lon2)
    Returns: distance in kilometers
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # radius of earth in kilometers
    return c * r
def get_query_weights(length, lambda_val=0.9):
    """
    Compute weights (the newer, the higher) for length N.
    Ex: w_i = lambda_val**(N-i-1)
    """
    return [lambda_val**(length - i - 1) for i in range(length)]
def weighted_dtw(query_seq, candidate_seq, query_weights):
    """
    Computes Weighted DTW distance between query_seq and candidate_seq
    """
    Nq = len(query_seq)
    Nc = len(candidate_seq)
    DTW = [[float('inf')] * (Nc + 1) for _ in range(Nq + 1)]
    DTW[0][0] = 0
    for i in range(1, Nq + 1):
        for j in range(1, Nc + 1):
            dist = haversine(query_seq[i-1], candidate_seq[j-1])
            cost = dist * query_weights[i-1]
            DTW[i][j] = min(
                DTW[i-1][j],       # skip candidate[j-1]
                DTW[i][j-1],       # skip query[i-1]
                DTW[i-1][j-1]      # align both
            ) + cost
    return DTW[Nq][Nc]

def batch_weighted_dtw(query_seq, candidate_seqs, lambda_val):
    """
    query_seq: List[Tuple[lat, lon]]
    candidate_seqs: List[List[Tuple[lat, lon]]]
    Returns: List of DTW distances (one for each candidate)
    """
    weights = get_query_weights(len(query_seq), lambda_val)
    scores = []
    for cand in candidate_seqs:
        score = weighted_dtw(query_seq, cand, weights)
        scores.append(score)
    return scores

def poiid_to_geocoord(poiid, city):
    """
    Convert POIID to geocoordinates.
    :param poiid: POI ID
    :param city: City name
    :return: (latitude, longitude)
    """
    # Load the POI metadata
    file_path = f"./raw_data/{city}_poi_mapping.csv"
    #  ,longitude,latitude
    # 0,103.964682,1.335005
    df = pd.read_csv(file_path)
    # Find the row as the POI ID is the index column. there is no POIID column, so we need to find the row by index
    row = df[df.index == poiid]
    # If the row is not found, raise an error
    if row.empty:
        raise ValueError(f"POIID {poiid} not found in {city} POI mapping file.")

    if not row.empty:
        longitude = row['longitude'].values[0]
        latitude = row['latitude'].values[0]
        return latitude, longitude
    else:
        raise ValueError(f"POIID {poiid} not found in {city} POI mapping file.")

def sequence_to_geocoords(sequence, city):
    """
    Convert a sequence of POIIDs to geocoordinates.
    :param sequence: List of POIIDs
    :param city: City name
    :return: List of tuples (latitude, longitude)
    """
    coords = []
    for poiid in sequence:
        try:
            coords.append(poiid_to_geocoord(poiid, city))
        except ValueError as e:
            print(e)
            coords.append((None, None))  # Append None if POIID not found
    return coords

def generate_sample_to_device(sample):
    sample_to_device = []
    if settings.enable_dynamic_day_length:
        last_day = sample[-1][5][0]
        for seq in sample:
            seq_day = seq[5][0]
            if last_day - seq_day < settings.sample_day_length:
                features = torch.tensor(seq[:5]).to(device)
                day_nums = torch.tensor(seq[5]).to(device)
                sample_to_device.append((features, day_nums))
    else:
        for seq in sample:
            features = torch.tensor(seq[:5]).to(device)
            day_nums = torch.tensor(seq[5]).to(device)
            sample_to_device.append((features, day_nums))

    return sample_to_device


def generate_day_sample_to_device(day_trajectory):
    features = torch.tensor(day_trajectory[:5]).to(device)
    day_nums = torch.tensor(day_trajectory[5]).to(device)
    day_to_device = (features, day_nums)
    return day_to_device


def train_model(train_set, test_set, vocab_size, device, save_name, rag_args=None):
    save_folder = "./experiments/"
    save_path = os.path.join(save_folder, save_name)
    print("save to: ", save_path)

    # create folder if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create predictions txt file
    predictions_file = os.path.join(save_path, "predictions_{}.txt".format(now_str))
    # create metrics csv file
    metrics_file = os.path.join(save_path, "metrics_{}.csv".format(now_str))

    # copy this python file to the save folder and add a now_str to the file name
    import shutil
    script_path = os.path.abspath(__file__)
    script_name = os.path.basename(script_path)
    new_script_name = f"{now_str}_{script_name}"
    new_script_path = os.path.join(save_path, new_script_name)
    shutil.copy(script_path, new_script_path)
    print("Script copied to: ", new_script_path)
    print("=============================================")


    torch.cuda.empty_cache()

    recall, ndcg, map = test_model(test_set, predictions_file, rag_args=rag_args)

    with open(metrics_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Recall', 'NDCG', 'MAP'])
        for k in recall.keys():
            writer.writerow([recall[k].item(), ndcg[k].item(), map[k].item()])
    print("Results saved to: ", metrics_file)
    print("=============================================")
    print("Recall: ", recall)
    print("NDCG: ", ndcg)
    print("MAP: ", map)
    print("=============================================")



def test_model(test_set, predictions_file, ks=[1, 5, 10], rag_args=None):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample_idx, sample in tqdm.tqdm(enumerate(test_set), desc="Testing", total=len(test_set)):

        # read the precictions file. if the sample_idx is already in the file, skip this sample
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                lines = f.readlines()
                if any(f"Sample {sample_idx}:" in line for line in lines):
                    print(f"Sample {sample_idx} already exists in {predictions_file}, skipping...")
                    continue

        sample_to_device = generate_sample_to_device(sample)

        first_rows_poi = [tensor[0] for tensor, _ in sample_to_device]
        first_rows_cate = [tensor[1] for tensor, _ in sample_to_device]
        longterm = torch.cat(first_rows_poi)[:-1].tolist()
        longterm_cate = torch.cat(first_rows_cate).tolist()
        recent = sample_to_device[-1][0][0][:-1].tolist()
        recent_cate = sample_to_device[-1][0][1][:-1].tolist()
        label = sample_to_device[-1][0][0][-1]

        ############RAG
        if rag_args.enable_rag:
            train_histories = load_list_of_lists("./experiments_rag/{}_train_short_term_history".format(rag_args.city))
            train_recommendations = load_list_of_lists("./experiments_rag/{}_train_short_term_recommendation".format(rag_args.city))
            valid_texts = [' '.join(map(str, recent))]
            train_texts = [' '.join(map(str, h)) for h in train_histories]
            embedder = Embedder(method='tfidf', device=device)
            embedder.fit(train_texts + valid_texts)
            train_embeds = embedder.encode(train_texts)
            valid_embeds = embedder.encode(valid_texts)
            retriever = Retriever(embedder=embedder, top_k=rag_args.top_k_rags, metric='cosine')
            topk_indices = retriever.retrieve_embed(valid_embeds, train_embeds)
            assert len(topk_indices) == 1
            topk_indices = topk_indices[0]
            topk_train_histories = [train_histories[i] for i in topk_indices]
            topk_train_recs = [train_recommendations[i] for i in topk_indices]
            # merge the topk_train_histories and topk_train_recs
            topk_train_histories_recs = []
            for i in range(len(topk_train_histories)):
                topk_train_histories[i].append(topk_train_recs[i])
                topk_train_histories_recs.append(topk_train_histories[i])

        ###############

        ############### GeoRerank
        if rag_args.enable_georerank:
            # convert the longterm and recent to geocoordinates
            recent_coords = sequence_to_geocoords(recent, rag_args.city)
            topk_train_histories_recs_coords = []
            for rec in topk_train_histories_recs:
                rec_coords = sequence_to_geocoords(rec, rag_args.city)
                topk_train_histories_recs_coords.append(rec_coords)

            # calculate the DTW distance
            dtw_scores = batch_weighted_dtw(recent_coords, topk_train_histories_recs_coords, lambda_val=rag_args.dtw_lambda_val)
            # sort the topk_train_histories_recs by the DTW distance
            sorted_indices = np.argsort(dtw_scores)
            # get the topk_train_histories_recs by the sorted indices
            topk_train_histories_recs = [topk_train_histories_recs[i] for i in sorted_indices]

        prompt = f"""\
<long-term check-ins> [Format: (POIID, category)]: {longterm}, {longterm_cate}
<recent check-ins> [Format: (POIID, category)]: {recent}, {recent_cate}
Your task is to recommend a user's next point-of-interest (POI) based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.
4. Do not include line breaks in your output. Do not recommend duplicate POIIDs.
5. Make sure the length of the recommendation list is exactly 10.

Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten most probable places in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). 
"""
        if rag_args.enable_rag:
            prompt = f"""\
<long-term check-ins> [Format: (POIID, category)]: {longterm}, {longterm_cate}
<recent check-ins> [Format: (POIID, category)]: {recent}, {recent_cate}
Your task is to recommend a user's next point-of-interest (POI) based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.
4. Do not include line breaks in your output. Do not recommend duplicate POIIDs.
5. Make sure the length of the recommendation list is exactly 10.

Context:
There are some <recent check-ins> [Format: (POIID)] by other users that might be useful: {topk_train_histories_recs}. 
You can decide by yourself whether to recommend them or not. 
Only use them as a reference when you are not sure what to recommend.

Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten most probable places in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements).
"""
            
        response = client.chat.completions.create(
        model=rag_args.llm,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": "You are a helpful assistant. Your task is to strictly follow the instructions."}
        ],
        temperature=0.0000001, # https://community.openai.com/t/clarifications-on-setting-temperature-0/886447/6
        timeout=100, 
        top_p=0,
        )
        response = response.choices[0].message.content
        # if response is covered by ```json and ```, remove it. also change two } to 1 }
        response =  response.replace("```json", "").replace("```", "").replace("}}", "}")

        if rag_args.enable_agenticragREAL:
            llm2_prompt = f"""\
    A user has completed the following task:
    Recommend the next ten POIs for a user based on their trajectory information, which includes a sequence of <long-term check-ins> [Format: (POIID, category)] and <recent check-ins> [Format: (POIID, category)] in chronological order.
    The elements in the format are as follows:
    - "POIID" refers to the unique id of the POI,
    - "Category" shows the semantic information of the POI.
    Here are the requirements for the task:
    1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
    2. Consider the recent check-ins to extract users' current preferences.
    3. Consider which "Category" the user would go next, as long-term check-ins indicate preferred sequential transitions.
    4. The response must not include line breaks. There must be no duplicate POIIDs in the recommendation.
    5. The recommendation list must have exactly 10 distinct POIIDs.
    Context:
    There are some <recent check-ins> [Format: (POIID)] by other users that might be useful: {topk_train_histories_recs}.
    They are provided for reference only. Use them only if needed and do not rely solely on this data.
    Here are the details for this user:
    <long-term check-ins> [Format: (POIID, category)]: {longterm}, {longterm_cate}
    <recent check-ins> [Format: (POIID, category)]: {recent}, {recent_cate}
    Here is the current answer from the user:
    {response}
    **Your task:**  
    - Carefully review the answer above.  
    - Evaluate whether it satisfies all requirements.  
    - If it is correct, concise, and well-formatted, reproduce the answer.  
    - If you find any issues (formatting, duplicates, insufficient recommendations, irrelevant POIIDs, unsatisfactory reasoning, etc), revise the answer to fully meet the requirements.
    - The result should be output in the same JSON object format as specified.
    Return ONLY the final (possibly revised) JSON object with the SAME format as the given answer, which is a JSON object containing your "recommendation" and "reason".
    """

            response = client.chat.completions.create(
            model=rag_args.llm,
            messages=[
                {"role": "user", "content": llm2_prompt},
                {"role": "system", "content": "You are a helpful assistant. Your task is to strictly follow the instructions."}
            ],
            temperature=0.0000001, # https://community.openai.com/t/clarifications-on-setting-temperature-0/886447/6
            timeout=100, 
            top_p=0,
            )
            response = response.choices[0].message.content
            response =  response.replace("```json", "").replace("```", "").replace("}}", "}")

        response_dict = json.loads(response)
        pred_l = []
        for POIID in response_dict["recommendation"]:
            # remove all potential letters, only keep numbers
            numbers_only = ''.join([char for char in str(POIID) if char.isdigit()])
            pred_l.append(int(numbers_only))
        pred = torch.tensor(pred_l)
        if len(pred) < 10:
            # warn
            print("Warning: prediction length is less than 10. Padding with random POIIDs.")
            # pad with random POIIDs
            random_poiids = random.sample(range(1, 1000), 10 - len(pred))
            pred = torch.cat((pred, torch.tensor(random_poiids)))

        # save sample_idx, pred, label to predictions_file
        with open(predictions_file, 'a') as f:
            f.write(f"Sample {sample_idx}: {pred}, {label}\n")

        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0).to(device)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1).to(device)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='RALLM_POI', help='Task name for saving')
    parser.add_argument('--city', type=str, default='PHO', help='City name')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--set', type=str, default='valid', help='Set name')
    parser.add_argument('--enable_rag', type=str2bool, nargs='?', const=True, default=True, help='Enable RAG')
    parser.add_argument('--enable_georerank', type=str2bool, nargs='?', const=True, default=True, help='Enable GeoRerank')
    parser.add_argument('--enable_agenticragREAL', type=str2bool, nargs='?', const=True, default=True, help='Enable AgenticRAGREAL')
    parser.add_argument('--dtw_lambda_val', type=float, default=0.8, help='Lambda value for DTW distance calculation')
    # top_k_rags
    parser.add_argument('--top_k_rags', type=int, default=10, help='Top K RAGs to retrieve')
    args = parser.parse_args()
    save_name = '{}_{}_{}_{}_rag{}_georerank{}_agentic{}_dtw{}_topkrags{}'.format(args.task_name, 
                                                                             args.city, 
                                                                             args.llm, 
                                                                             args.set, 
                                                                             args.enable_rag, 
                                                                             args.enable_georerank, 
                                                                             args.enable_agenticragREAL,
                                                                             args.dtw_lambda_val,
                                                                             args.top_k_rags)

    #
    misc_args_dict = {}
    #
    city = args.city
    run_times = 1
    
    device = settings.gpuId if torch.cuda.is_available() else 'cpu'

    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now: ", now_str)

    processed_data_directory = './processed_data/original'

    # Read training data
    file = open(f"{processed_data_directory}/{city}_train", 'rb')
    train_set = pickle.load(file)
    # inspect the train set
    print("==============Example train set===============")
    print(train_set[0])
    file = open(f"{processed_data_directory}/{city}_valid", 'rb')
    valid_set = pickle.load(file)
    file = open(f"{processed_data_directory}/{city}_test", 'rb')
    test_set = pickle.load(file)

    # Read meta data
    file = open(f"{processed_data_directory}/{city}_meta", 'rb')
    meta = pickle.load(file)
    print("========Length of meta data and examples==========")
    print("\nPOI:", len(meta["POI"]), "\nFirst 5 POIs:", meta["POI"][:5])
    print("\ncat:", len(meta["cat"]), "\nFirst 5 categories:", meta["cat"][:5])
    print("\nuser:", len(meta["user"]), "\nFirst 5 users:", meta["user"][:5])
    print("\nhour:", len(meta["hour"]), "\nFirst 5 hours:", meta["hour"][:5])
    print("\nday:", len(meta["day"]), "\nFirst 5 days:", meta["day"][:5])
    print("==================================================")
    # print(len(meta["POI"]), len(meta["cat"]), len(meta["user"]), len(meta["hour"]), len(meta["day"]))
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}


    print(f'Current GPU {settings.gpuId}')
    run_num = 0

    if args.set == 'valid':
        INFER_SET = valid_set
    elif args.set == 'test':
        INFER_SET = test_set
    elif args.set == 'validtest':
        INFER_SET = valid_set + test_set
    else:
        raise ValueError("Invalid set name. Please choose 'valid' or 'test'.")
    train_model(train_set, INFER_SET, vocab_size, device, save_name, rag_args=args)