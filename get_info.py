import pickle
import json
import random
import sys
import numpy as np
def get_random_proof_0_4():
    file_path_data_combined = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_20w_v3_rand/data_5_vars/_v3.json'
    train_key_path = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/key_directory/key_20w_quantile_0_66_0_8_train.json'

    train_key = json.load(open(train_key_path, 'r'))
    data = json.load(open(file_path_data_combined, 'r'))
    print(len(data.keys()))
    print(len(train_key))
    ith_shortest = 3
    # 39799 no solution appears

    train_data = []
    for key in train_key:
        length_list_per_theorem = []
        for solution in data[key]:
            length_list_per_theorem.append([len(solution['input'].split()) + len(solution['output'].split()), solution])

        random_numbers_to_be_picked = random.sample(range(5), 2)

        length_list_per_theorem = sorted(length_list_per_theorem, key=lambda x: x[0])
        temp_dict = {}
        temp_dict['source'] = length_list_per_theorem[random_numbers_to_be_picked[0]][1]['input']
        temp_dict['target'] = length_list_per_theorem[random_numbers_to_be_picked[0]][1]['output']
        train_data.append(temp_dict)

        temp_dict_1 = {}
        temp_dict_1['source'] = length_list_per_theorem[random_numbers_to_be_picked[1]][1]['input']
        temp_dict_1['target'] = length_list_per_theorem[random_numbers_to_be_picked[1]][1]['output']
        train_data.append(temp_dict_1)

    print(len(train_data))
    print(train_data[100]['source'])
    print(train_data[100]['target'])

    json.dump(train_data, open(
        '/home/zihan/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_rand_0_4/train_val_test/train_data.json',
        'w'))

    temp_list = json.load(open(
        '/home/zihan/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_rand_0_4/train_val_test/train_data.json',
        'r'))

    print(len(temp_list))
def check_proof_length():
    key_list = list(info.keys())[:-1]
    generated_solution_length_list = []
    standard_solution_length_list = []
    counter = 0
    for key in key_list:
        if info[key]['proof_is_success'] == True:
            length_gen = len(info[key]['generated_solution'].split())
            length_ori = len(data_combined[key]['v_3_2_solution']['input'].split()) + len(
                data_combined[key]['v_3_2_solution']['output'].split())
            generated_solution_length_list.append(length_gen)
            standard_solution_length_list.append(length_ori)

            if length_ori - 500 > length_gen:
                counter += 1
                if counter <= 10:
                    print('-------------------------------------')
                    print(f'Num{counter} with key {key}-----------------')
                    print('original solution:')
                    print(data_combined[key]['v_3_2_solution']['output'])
                    print('-------------------------------------')
                    print('generated solution:')
                    print(info[key]['proof'])

                else:
                    sys.exit()

    print(f'generated solution mean length for correct proof is {np.mean(generated_solution_length_list)}')
    print(f'original solution mean length for correct proof is {np.mean(standard_solution_length_list)}')

    sys.exit()

def compare_proof_length():
    key_list_v3_rand = list(info_v3_rand.keys())[:-1]
    key_list_DFS = list(info_DFS['key_final_state'].keys())

    both_true = []
    for key in key_list_DFS:
        if info_DFS['key_final_state'][key] == 'success' :
            both_true.append(key)

    length_DFS = []
    length_v3_rand = []
    for key in both_true:
        length_DFS.append(len(info_DFS['key_proof'][key].split('\n')))
        #length_v3_rand.append(len(info_v3_rand[key]['proof'].split('\n')))

    print(f'same true is {len(both_true)}')
    print(f"DFS {np.mean(length_DFS)}")
    #print(f"v3_rand {np.mean(length_v3_rand)}")

def compare_length_of_tactics():
    file_path = '/home/c5an/leandojo_project/atp_research/DFS/output/test_ge_0_80_1027_steps_trained_le_0_66_20w_v_3_rand_0_4_checkpoint_3300_max_output_1500_v3_le_1200.pkl'
    file_path_DFS = '/home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_100_num_sampled_15_temp_1_5_1000_checkpoint_1600.pkl'


    info_trialmaster = pickle.load(open(file_path,'rb'))
    info_DFS = pickle.load(open(file_path_DFS,'rb'))
    print('info_trialmaster', info_trialmaster['stats'])
    #print('info_DFS', info_DFS['stats'])
    #print(info_DFS.keys())


    sys.exit()
    #print(info_DFS.keys())
    #print('-------')
    #print(info_trialmaster.keys())
    #sys.exit()
    key_list = list(info_trialmaster.keys())[2:-2]


    count_DFS = 0
    count_trialmaster = 0
    total_count_DFS = 0
    total_count_trialmaster = 0
    for key in key_list:
        #print('what is key',key)

        #print(info_DFS['key_proof'][key])
        #print(info_trialmaster[key]['proof'])
        #print(info_DFS['key_final_state'][key])
        #print(info_trialmaster[key]['proof_is_success'])
        #sys.exit()
        if info_DFS['key_final_state'][key] != 'success' or info_trialmaster[key]['proof_is_success'] != True:
            continue
        string_DFS = info_DFS['key_proof'][key].split('\n')[3:]
        string_trialmaster = info_trialmaster[key]['proof'].split('\n')[3:]

        for ele in string_DFS:
            total_count_DFS += len(ele)
            #print(len(ele))
            count_DFS += 1
        #print('------')
        for ele in string_trialmaster:
            total_count_trialmaster += len(ele)
            #print(len(ele))
            count_trialmaster += 1

    print(float(total_count_DFS)/float(count_DFS))
    print(float(total_count_trialmaster)/float(count_trialmaster))

def count_backtrack():
    file_path = '/home/c5an/leandojo_project/atp_research/DFS/output/new_no_failed_path_500.pkl'
    file_path_DFS = '/home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_100_num_sampled_15_temp_1_5_1000_checkpoint_1600.pkl'

    info_trialmaster = pickle.load(open(file_path, 'rb'))
    info_DFS = pickle.load(open(file_path_DFS, 'rb'))
    print('info_trialmaster', info_trialmaster['stats'])
    # print('info_DFS', info_DFS['stats'])
    # print(info_DFS.keys())
    sys.exit()
    # print(info_DFS.keys())
    # print('-------')
    # print(info_trialmaster.keys())
    # sys.exit()
    key_list = list(info_trialmaster.keys())[2:-2]

    count_backtrack = 0
    count_what_so_ever = 0
    for key in key_list:
        if info_trialmaster[key]['proof_is_success'] == False:
            continue
        count_what_so_ever += 1

        if 'no solution' in info_trialmaster[key]['generated_solution']:
            count_backtrack += 1

    print(count_what_so_ever, count_backtrack)
if __name__ == '__main__':
    path_v3_rand = 'DFS/output/test_ge_0_80_1027_steps_trained_le_0_66_20w_v_3_rand_0_4_checkpoint_3300_max_output_1500_v3_le_1200.pkl'
    #path = 'output/outcome_basic_key_ge_080_100_num_sampled_20_temp_2_0_1000_checkpoint_1600.pkl'
    path_DFS = "DFS/output/outcome_basic_key_ge_080_100_num_sampled_10_temp_1_8_1000_checkpoint_1600.pkl"
    path_DFS = "DFS/output/outcome_basic_key_ge_080_100_num_sampled_10_temp_1_5_1000.pkl"
    path_DFS = "DFS/output/output_066_08_no_failed_path_cp_8100.pkl"
    file_path = 'output/new_no_failed_path_500.pkl'
    file_path = 'output/new_no_failed_path_500.pkl'

    file_path = '/home/c5an/leandojo_project/atp_research/DFS/output/new_no_failed_path_500.pkl'
    #file_path_DFS = '/home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_100_num_sampled_15_temp_1_5_1000_checkpoint_1600.pkl'
    file_path_DFS = '/home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_100_num_sampled_10_temp_1_8_1000_checkpoint_1600.pkl'
    info_trialmaster = pickle.load(open(file_path,'rb'))
    #info_DFS = pickle.load(open(file_path_DFS,'rb'))

    print('info_trialmaster', info_trialmaster['stats'])
    #print('info_DFS', info_DFS['stats'])
    #print(info_DFS.keys())
    sys.exit()
    #print(info_DFS.keys())
    #print('-------')
    #print(info_trialmaster.keys())
    #sys.exit()
    key_list = list(info_trialmaster.keys())[2:-2]


    count = 0
    for key in key_list:
        if info_trialmaster[key]['proof_is_success'] == True and info_DFS['key_final_state'][key] != 'success' and \
                'no solution' in info_trialmaster[key]['generated_solution']:
            count += 1

    print(count)

    sys.exit()

    key_list = list(info_DFS['key_final_state'].keys())
    counter = 0
    correct_list = []
    for key in key_list:
        print(info_DFS['key_final_state'][key])
        if info_DFS['key_final_state'][key] == 'success':
            correct_list.append(key)
            counter += 1
    print(counter)

    v3_base_correct = json.load(open('/home/c5an/v3_base_correct.json','r'))

    list1 = correct_list
    list2 = v3_base_correct

    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)

    # Calculate the quantity of elements in list1 but not in list2
    quantity_in_list1_not_in_list2 = len(set1 - set2)

    # Calculate the quantity of elements in list2 but not in list1
    quantity_in_list2_not_in_list1 = len(set2 - set1)

    print("Quantity of elements in correct_list but not in v3_base_correct:", quantity_in_list1_not_in_list2)
    print("Quantity of elements in v3_base_correct but not in correct_list:", quantity_in_list2_not_in_list1)



    '''outcome['stats'] = {}
        outcome['stats']['total_lean_count_single_backtrack'] = count_lean_single_backtrack
        outcome['stats']['total_lean_count_multiple_backtrack'] = count_lean_multiple_backtrack
        outcome['stats']['num_success'] = self.counter_success
        outcome['stats']['num_failed'] = self.counter_failed
        outcome['stats']['num_too_long'] = self.counter_too_long
        outcome['stats']['num_sampled_tactics'] = self.num_sampled_tactics
        outcome['stats']['temperature'] = self.temperature
        outcome['key_final_state'] = self.root
        outcome['key_lean_count'] = self.count_lean_dict
        outcome['key_proof'] = proof_dict
        outcome['tactic_list_tree'] = self.tactic_list_tree'''

    '''dict_keys(['237995755978852082346969412535', 'initial_state', 'generated_solution', 'steps', 'error_info', 'error_output', 'error_context', 
    'proof', 'proof_is_too_long', 'proof_is_success', 'num_no_solution', 'tactic_state_list', 'count_tactic_lean'])'''

    #compare_proof_length()

    #print(inf
    # o['key_lean_count']['84582691312694602867768734711'])

    sys.exit()
    proof_dict = info['key_proof']
    for key in proof_dict.keys():
        print(proof_dict[key])
        break
    #root_dict = info['key_final_state']
'''    counter = 0
    in_process_key_list = []
    for key in root_dict.keys():
        if root_dict[key] == 'open':
            counter += 1
            in_process_key_list.append(key)

    print(len(in_process_key_list))

    print(counter)

    saved_path = '/home/c5an/leandojo_project/atp_research/DFS/key_directory/key_in_process_round_65_nst_20_temp_2_0.json'
    json.dump(in_process_key_list,open(saved_path,'w'))'''




