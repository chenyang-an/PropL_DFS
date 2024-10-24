import json
import pickle
import numpy as np
import sys
# PYTHONPATH=/home/c5an/leandojo_project/atp_research python -m leandojo_project.atp_research.DFS.get_info_previous

def check_lean_steps():
    key_list = list(info.keys())[:-1]
    proof_step_list = []
    for key in key_list:
        proof_step_list.append(info[key]['steps'])


    print(f'average for {version} solution call lean number', np.mean(proof_step_list))
    print(f'25 quantile for {version} solution call lean number', np.quantile(proof_step_list,0.25))
    print(f'50 quantile for {version} solution call lean number',np.quantile(proof_step_list,0.5))
    print(f'75 quantile for {version} solution call lean number',np.quantile(proof_step_list,0.75))
    print(f'total for {version} solution call lean number', np.sum(proof_step_list))

    correct_proof_step_list = []
    for key in key_list:
        if info[key]['proof_is_success'] == True:
            correct_proof_step_list.append(info[key]['steps'])

    print(f'average for correct {version} solution call lean number', np.mean(correct_proof_step_list))
    print(f'25 quantile for correct {version} solution call lean number', np.quantile(correct_proof_step_list, 0.25))
    print(f'50 quantile for correct {version} solution call lean number', np.quantile(correct_proof_step_list, 0.5))
    print(f'75 quantile for correct {version} solution call lean number', np.quantile(correct_proof_step_list, 0.75))
    print(f'total for correct {version} solution call lean number', np.sum(correct_proof_step_list))

def check_proof_length():
    key_list = list(info.keys())[:-1]
    generated_solution_length_list = []
    standard_solution_length_list = 0
    for key in key_list:
        if info[key]['proof_is_success'] == True:
            generated_solution_length_list.append(len(info[key]['generated_solution'].split()))
            standard_solution_length_list.append(
                len(data_combined[key]['v_3_2_solution']['input']) + len(data_combined[key]['v_3_solution']['output']))
    print(f'generated solution mean length for correct proof is {np.mean(generated_solution_length_list)}')
    print(f'original solution mean length for correct proof is {np.mean(standard_solution_length_list)}')


if __name__ == '__main__':
    file_path_data_combined = "/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/data_combined.json"

    #data_combined = json.load(open(file_path_data_combined,'r'))

    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v2_max_200/output/test_le_400_ge_300_steps_trained_le_200_20w_v2_0_46epo_checkpoint_2200_max_output_2000.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v2_max_200/output/test_le_400_ge_300_steps_trained_le_200_20w_v2_0_46epo_checkpoint_800_max_output_2500.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v2_max_200/output/test_le_200_steps_trained_le_200_20w_v2_0_46epo_checkpoint_800_max_output_2500.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v2_max_200/output/test_le_400_ge_300_steps_trained_le_200_20w_v2_0_46epo_checkpoint_1400_max_output_2500.pkl'
    file_path = "/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v_3_max_400_based_on_v3_2/output/test_le_1200_ge_600_steps_trained_le_400_20w_v_3_0_38epo_checkpoint_1700_max_output_1500_v3_le_1200.pkl"
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v_3_2_max_400/output/test_le_1200_ge_600_steps_trained_le_400_20w_v_3_2_0_42epo_checkpoint_1900_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v_3_2_max_400/output/test_le_400_steps_trained_le_400_20w_v_3_2_0_42epo_checkpoint_1900_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v_3_max_400_based_on_v3_2/output/test_le_400_steps_trained_le_400_20w_v_3_0_38epo_checkpoint_1700_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_n_16_v_3_2_max_400/output/test_le_1200_ge_600_steps_trained_le_400_20w_v_3_2_0_42epo_checkpoint_1900_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_2/output/test_ge_0_80_678_steps_trained_le_0_66_335_20w_v_3_2_checkpoint_2100_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_2/output/test_le_0_66_335_steps_trained_le_0_66_335_20w_v_3_2_checkpoint_1600_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_2/output_true/test_ge_0_80_1027_steps_trained_le_0_66_609_20w_v_3_2_checkpoint_1600_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3_2/output_true/test_ge_0_80_1027_steps_trained_le_0_66_609_20w_v_3_2_checkpoint_1600_max_output_1500_v3_le_1200.pkl'
    file_path = '/home/c5an/train_scripts/train_scripts_prop_serv8/train_Llama_7b_PropL_20w_0_66_0_88_v3/output/test_ge_0_80_1027_steps_trained_le_0_66_609_20w_v_3_checkpoint_1600_max_output_1500_v3_le_1200.pkl'

    version = 'v3'
    print(f'file_path is {file_path}')
    #data_combined = json.load(open(file_path_data_combined,'r'))
    info = pickle.load(open(file_path, 'rb'))
    """{'proof_success': 593, 'total_theorem': 1000, 'proof_error': 406, 'over_1500': 1, 'model_checkpoint': '/home/c5an/train_scripts/train_scripts_prop_serv8/v3_2_0_66_0_8/checkpoint-1600', 'test_data_path': '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/key_directory/key_20w_quantile_0_66_0_8_out_dist_test.json'}
    dict_keys(['84582691312694602867768734711', 'initial_state', 'generated_solution', 'steps', 'error_info', 'error_output', 'error_context', 'proof', 'proof_is_too_long', 'proof_is_success', 'num_no_solution', 'tactic_state_list'])"""
    print(info['stats'])
    key_list = list(info.keys())[:-1]
    counter = 0
    correct_key_list = []
    for key in key_list:
        print(info[key].keys())
        if info[key]['proof_is_success'] == True:
            counter += 1
            correct_key_list.append(key)


    print(counter)
    json.dump(correct_key_list,open('/home/c5an/v3_base_correct.json','w'))









    sys.exit()
    #check_lean_steps()
    key_list = list(info.keys())[:-1]
    generated_solution_length_list = []
    standard_solution_length_list = []
    counter = 0
    for key in key_list:
        if info[key]['proof_is_success'] == True:
            length_gen = len(info[key]['generated_solution'].split())
            length_ori = len(data_combined[key]['v_3_2_solution']['input'].split())+len(data_combined[key]['v_3_2_solution']['output'].split())
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
    '''dict[key]['v_1_solution'] = data_v1[key]
    dict[key]['v_2_solution'] = data_v2[key]
    dict[key]['v_3_solution'] = data_v3[key]
    dict[key]['v_3_2_solution'] = data_v3_2[key]
    dict[key]['num_nodes'] = nodes[key]'''

    '''dict_keys(
        ['64842255541951174146418410033', 'initial_state', 'generated_solution', 'steps', 'error_info', 'error_output',
         'error_context', 'proof', 'proof_is_too_long', 'proof_is_success', 'num_no_solution', 'tactic_state_list'])'''

    counter = 0
    version = 'v_3_2'
    for key in key_list:
        original_solution_to_be_compared = data_combined[key][f'{version}_solution']['input'] + data_combined[key][f'{version}_solution']['output'] + '\n'
        if info[key]['proof_is_success'] == True:
            if len(original_solution_to_be_compared.split()) - len(info[key]['generated_solution'].split()) > 350:
                counter += 1
                print('---------original')
                print(original_solution_to_be_compared)
                print('---------generated')
                print(info[key]['generated_solution'])

    print(counter)



    sys.exit()
    version = 'v_3'
    length_ge = []
    length_original_solution = []
    for key in key_list:
        original_solution_to_be_compared = data_combined[key][f'{version}_solution']['input'] + data_combined[key][f'{version}_solution']['output'] + '\n'
        if info[key]['proof_is_success'] == True:
            length_ge.append(len(info[key]['generated_solution'].split()))
            length_original_solution.append(len(original_solution_to_be_compared.split()))

            '''if len(info[key]['generated_solution'].split()) < (len(v_2_solution_to_be_compared.split()) - 100):
                counter_1 += 1
                print(f"key is {key}")
                print("Generated_solution-------")
                print(info[key]['generated_solution'])
                print("Original_solution-------")
                print(v_2_solution_to_be_compared)
                print()
                print()
                print()
            else:
                pass
'''


    print(np.mean(length_ge))
    print(np.mean(length_original_solution))
    print(np.std(length_ge))
    print(np.std(length_original_solution))




