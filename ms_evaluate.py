from propositional_logic.random_gen.evaluation_access import *
from vllm import LLM, SamplingParams
import time
import json
import pickle
import os
import time
import re
import sys
from tqdm import tqdm
import argparse
from loguru import logger
from tqdm import tqdm
import concurrent.futures
import random
import subprocess

import uuid


# python -m DFS.ms_evaluate /home/c5an/leandojo_project/atp_research/DFS/ms_generation/propl-cot-llama-7b-4gpu-new-checkpoint-12500_generation.pkl
class Prover:
    def __init__(
            self,
            test_theorem_list,
            generation_file_path,
            max_workers,
            saved_file_path,
            experiment_id
    ) -> None:
        self.max_workers = max_workers
        self.test_theorem_list = test_theorem_list
        self.count_lean_dict = {}
        self.counter_success = 0
        self.counter_failed = 0
        self.counter_in_process = 0
        self.counter_too_long = 0
        self.theorem_object_dict = {}
        self.prompts_tactic_state_list = {}
        self.root = {}
        self.proof_generation = {}
        self.parent_node_of_node = {}
        self.round = 0
        self.saved_file_path = saved_file_path
        self.experiment_id = experiment_id
        self.key_to_be_infered = []
        self.key_not_finished = []
        self.prompts_entered = []
        self.counter_failed_with_error = 0
        self.tactic_dict = {}
        self.generation_file_path = generation_file_path
        generation_file = pickle.load(open(generation_file_path,'rb'))

        print(f"{len(self.test_theorem_list)} many theorem loaded")
        for key in tqdm(self.test_theorem_list):
            sample_eval = SingleTheoremEval(5, int(key))
            self.theorem_object_dict[key] = sample_eval
            init_state = self.theorem_object_dict[key].get_initial_prompt()
            #print('init_state is ', init_state)
            self.root[key] = 'open'  # open, success or failed
            self.proof_generation[key] = True
            #generation_file[str(key)].reverse()
            self.tactic_dict[key] = generation_file[str(key)]
            #print(self.tactic_dict[key])
            #sys.exit()
        print('initialization done')
    def get_current_state_number(self, key):
        string = self.theorem_object_dict[key].get_current_state_with_label()
        for line in string.split('\n'):
            break
        return line

    def search(self):
        while True:
            print(f'Round {self.round}------')

            self.key_not_finished = []
            key_prompt_length_list = []
            for key in tqdm(self.test_theorem_list, total=len(self.test_theorem_list),
                            desc=f"check state for theorems for round {self.round}"):
                if self.root[key] == 'open' and self.round < len(self.tactic_dict[key]):
                    self.key_not_finished.append(key)

            #print(f'key to be infered is {self.key_not_finished}')
            if len(self.key_not_finished) == 0:
                print('test finished')
                self.save_outcome()
                break


            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                list(tqdm(executor.map(self.search_per_key_per_step, self.key_not_finished),
                          total=len(self.key_not_finished), desc=f"Lean_verifying for Round {self.round}"))

            for key in tqdm(self.key_not_finished, total=len(self.key_not_finished),
                            desc=f"Retrieve results from concurrent processing for Round {self.round}"):
                report_key_file = pickle.load(open(
                    f'/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file_{key}_{self.round}_{self.experiment_id}.pkl',
                    'rb'))
                self.root[key] = report_key_file['key_status']
                self.theorem_object_dict[key] = report_key_file['theorem_object_dict']
                self.proof_generation[key] = report_key_file['proof_generation']

            self.round += 1

    def search_per_key_per_step(self, key):
        # print(f'key is {key}, current tactic_list_tree for Round {self.round} is {self.tactic_list_tree[key]}')
        key_status = 'open'
        #print('entered_tactic is ', entered_tactic)
        try:
            entered_tactic = (f'state_{self.round}_tactic_0:', self.tactic_dict[key][self.round][0])
            lean_output = self.theorem_object_dict[key].provide_tactic(entered_tactic[0], entered_tactic[1])

            ground_truth_state = lean_output[1]
            ground_truth_state_list = ground_truth_state.split('\n')[1:]
            ground_truth_state_revised = ''.join(ground_truth_state_list)
            ground_truth_state_revised = ''.join(ground_truth_state_revised.split())

            generated_state = self.tactic_dict[key][self.round][1]
            generated_state_list = generated_state.split('\n')
            generated_state_revised = ''.join(generated_state_list)
            generated_state_revised = ''.join(generated_state_revised.split())

            #print(ground_truth_state_revised)
            #print(generated_state_revised)

            if ground_truth_state_revised == generated_state_revised:
                if ground_truth_state_revised != 'nogoalsproofiscomplete':
                    print('state is true', ground_truth_state_revised)
            else:
                self.proof_generation[key] = False
            #print(ground_truth_state_revised==generated_state_revised)

            if 'proof is complete' in lean_output[1]:
                key_status = 'success'
                #print(f'key is {key}, search is success!')
                #print(f'key is {key}, the successful proof \n{self.theorem_object_dict[key].get_current_lean_proof()}')

        except Exception as e:
            #print(f"key is {key}, exception happend")
            #print(e)
            key_status = 'failed'
            self.proof_generation[key] = False

        finally:
            report_key_file = {}
            report_key_file['key_status'] = key_status
            report_key_file['theorem_object_dict'] = self.theorem_object_dict[key]
            report_key_file['proof_generation'] = self.proof_generation[key]
            pickle.dump(report_key_file, open(
                f'/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file_{key}_{self.round}_{self.experiment_id}.pkl',
                'wb'))

    def save_outcome(self):
        open_count = 0
        fail_count = 0
        success_count = 0
        all_correct_count = 0
        for key in tqdm(self.test_theorem_list, total=len(self.test_theorem_list),
                        desc=f"check state for theorems for round {self.round}"):
            if self.root[key] == 'open':
                open_count += 1
            if self.root[key] == 'failed':
                fail_count += 1
            if self.root[key] == 'success':
                success_count += 1
            if self.proof_generation[key] == True:
                all_correct_count += 1
        print('open_count is ', open_count)
        print('fail_count is ', fail_count)
        print('success_count is ', success_count)
        print('all correct count is ', all_correct_count)
        print('generation file path is ', self.generation_file_path)

if __name__ == '__main__':
    '''os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if os.environ.get("TOKENIZERS_PARALLELISM") == "true":
        print("TOKENIZERS_PARALLELISM is set to true")
    else:
        print("TOKENIZERS_PARALLELISM is not set to true")'''

    parser = argparse.ArgumentParser(description='Description of your program.')
    #parser.add_argument('test_data_path', type=str, help='test_data_path')
    #parser.add_argument('saved_file_path', type=str, help='test_data_path')
    #parser.add_argument('max_workers', type=str, help='test_data_path')
    parser.add_argument('generation_file_path', type=str, help='generation_file_path')

    args = parser.parse_args()
    generation_file_path = args.generation_file_path
    '''test_data_path = args.test_data_path
    saved_file_path = args.saved_file_path
    max_workers = args.max_workers'''

    swap_space = 100
    max_workers = 20
    num_tested_theorem = 1000
    test_data_path = '/home/c5an/leandojo_project/atp_research/DFS/ms_generation/test_key_list_6_7_8.json'

    saved_file_path = '/home/c5an/leandojo_project/atp_research/ms_result'
    print(f'test_data_path is {test_data_path}')
    print(f'generation_file_path is {generation_file_path}')
    print(f'saved_file_path is {saved_file_path}')
    print(f'max_workers is {max_workers}')

    print(f'swap_space is {swap_space}')
    with open(test_data_path, 'r') as f:
        test_theorem_list = json.load(f)
    print(f'length of test theorem list is {len(test_theorem_list)}')

    test_theorem_list = test_theorem_list[:num_tested_theorem]
    experiment_id = uuid.uuid4()  # Generates a random UUID.
    print('experiment id is ', experiment_id)

    evaluate_obj = Prover(test_theorem_list=test_theorem_list, max_workers=int(max_workers),
                       saved_file_path=saved_file_path, experiment_id=experiment_id, generation_file_path=generation_file_path)

    evaluate_obj.search()
    print('Now we start saving')

    print('Now we finish saving. exit')
    # command = "rm DFS/temp/temp_file*"
    # result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)






    '''
python -m DFS.ms_evaluate /home/c5an/leandojo_project/atp_research/DFS/ms_generation/propl-cot-llama-7b-4gpu-new-checkpoint-2500_generation.pkl
python -m DFS.ms_evaluate /home/c5an/leandojo_project/atp_research/DFS/ms_generation/propl-cot-llama-7b-4gpu-new-checkpoint-5000_generation.pkl
python -m DFS.ms_evaluate /home/c5an/leandojo_project/atp_research/DFS/ms_generation/propl-cot-llama-7b-4gpu-new-checkpoint-10000_generation.pkl
python -m DFS.ms_evaluate /home/c5an/leandojo_project/atp_research/DFS/ms_generation/propl-cot-llama-7b-4gpu-new-checkpoint-12500_generation.pkl

'''