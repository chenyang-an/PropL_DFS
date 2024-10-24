import json
import pickle
import re
import sys
#PYTHONPATH=/home/c5an/leandojo_project/atp_research python -m leandojo_project.atp_research.DFS.upload_model
file_path = '/home/c5an/leandojo_project/atp_research/DFS/output/test_ge_0_80_1027_steps_trained_le_0_66_20w_v_3_rand_0_4_checkpoint_3300_max_output_1500_v3_le_1200.pkl'
file_path_DFS = '/home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_100_num_sampled_15_temp_1_5_1000_checkpoint_1600.pkl'
file_ood_keys = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/key_directory/key_20w_quantile_0_66_0_8_out_dist_test.json'
file_v2 = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/_v2.json'
info_trialmaster = pickle.load(open(file_path, 'rb'))
info_DFS = pickle.load(open(file_path_DFS, 'rb'))
key_ood = json.load(open(file_ood_keys,'r'))
dict = json.load(open(file_v2,'r'))
# print('info_trialmaster', info_trialmaster['stats'])
# print('info_DFS', info_DFS['stats'])
# print(info_DFS.keys())

# print(info_DFS.keys())
# print('-------')
# print(info_trialmaster.keys())
# sys.exit()



key_list = list(info_trialmaster.keys())[0:-1]

print(len(key_list))

count_total_lean_queries = 0
for key in key_list:
    #print(key)
    pattern = r'state_(\d+)_tactic_(\d+):'
    for line in dict[key]['output'].split('\n'):
        if re.findall(pattern, line):
            count_total_lean_queries += 1

print(count_total_lean_queries)

