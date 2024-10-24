import json
import numpy as np

import sys


def process_solution(string):
    string_list = string.split('\n')
    tactic_list = []
    for idx, line in enumerate(string_list):
        if 'tactic' in line:
            tactic_list.append(string_list[idx:idx+2])
        elif 'no solution' in line:
            tactic_list.append(string_list[idx])
        else:
            pass
    return tactic_list





if __name__ == '__main__':
    #file_path_v1 = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_original/data_5_vars/_v1.json'

    answer_list = json.load(open('/home/c5an/leandojo_project/atp_research/DFS/answer_test_key.json','r'))
    print(len(answer_list.keys()))

    print(answer_list[f'{list(answer_list.keys())[0]}'])

    sys.exit()



    file_path_data_combined = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/data_combined.json'
    test_data_path = '/home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/key_directory/key_20w_quantile_0_66_0_8_out_dist_test.json'
    with open(test_data_path, 'r') as f:
        test_theorem_list = json.load(f)

    data_combined = json.load(open(file_path_data_combined,'r'))

#v2 length generalization test: train on le 150

    tactic_list_dict = {}
    print(len(test_theorem_list))
    for key in test_theorem_list:
        #print(data_combined[key]['v_3_solution']['output'])
        #print('----------')
        tactic_list = process_solution(data_combined[key]['v_3_solution']['output'])
        tactic_list_dict[key]= tactic_list

    json.dump(tactic_list_dict,open('/home/c5an/leandojo_project/atp_research/DFS/answer_test_key.json','w'))



