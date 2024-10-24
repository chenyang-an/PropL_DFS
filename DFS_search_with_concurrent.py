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





#python -m DFS.DFS_search_with_concurrent /home/c5an/train_scripts/train_scripts_prop_serv8/v3_2_0_66_0_8/checkpoint-1600 1 /home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_45w/data_5_vars/key_directory/key_20w_quantile_0_66_0_8_out_dist_test.json /home/c5an/leandojo_project/atp_research/DFS/output/outcome_basic_key_ge_080_1000.pkl

class DFS:
    def __init__(
        self,
        num_sampled_tactics: int,
        temperature,
        test_theorem_list,
        max_workers,
        saved_file_path,
        experiment_id
    ) -> None:
        self.max_workers = max_workers
        self.test_theorem_list = test_theorem_list
        self.whether_backtrack = {}
        self.num_sampled_tactics = num_sampled_tactics
        self.count_lean_dict = {}
        self.counter_success = 0
        self.counter_failed = 0
        self.counter_in_process = 0
        self.counter_too_long = 0
        self.tactic_list_tree = {}
        self.theorem_object_dict = {}
        self.prompts_tactic_state_list = {}
        self.root = {}
        self.round_count = {}
        self.parent_node_of_node = {}
        self.round = 0
        self.saved_file_path = saved_file_path
        self.cstate_round = {} #should never be used once we want to apply tactic. should only be used at the beginning of each round
        self.experiment_id = experiment_id
        self.key_to_be_infered = []
        self.key_not_finished = []
        self.prompts_entered = []
        self.counter_failed_with_error = 0
        self.temperature = temperature

        print(f"{len(self.test_theorem_list)} many theorem loaded")
        for key in tqdm(self.test_theorem_list):
            sample_eval = SingleTheoremEval(5, int(key))
            self.theorem_object_dict[key] = sample_eval
            init_state = self.theorem_object_dict[key].get_initial_prompt()

            self.parent_node_of_node[key] = {}
            self.prompts_tactic_state_list[key] = [init_state]
            self.root[key] = 'open' #open, success or failed
            self.tactic_list_tree[key] = {}
            self.tactic_list_tree[key]["state_0:"] = None
            self.parent_node_of_node[key]["state_0:"] = None
            self.count_lean_dict[key] = {}
            self.count_lean_dict[key]['count_lean_multiple_backtrack'] = 0
            self.count_lean_dict[key]['count_lean_single_backtrack'] = 0
            self.count_lean_dict[key]['count_lean_tactic_success'] = 0
            self.round_count[key] = 0
            self.whether_backtrack[key] = False

        print('initialization done')
    def get_current_state_number(self, key):
        string = self.theorem_object_dict[key].get_current_state_with_label()
        for line in string.split('\n'):
            break
        return line
    def get_prev_state_number(self, key):
        string = self.theorem_object_dict[key].get_prev_state_with_label()
        for line in string.split('\n'):
            break
        return line
    def revise_entered_tactic(self,entered_tactic,key):
        if len(entered_tactic) != 2:
            assert True==False
        current_state_label = self.get_current_state_number(key)
        entered_tactic[0] = current_state_label[:-1] + '_tactic_0:'
        return entered_tactic
    def back_track_tactic(self, key):
        current_state_number = self.get_current_state_number(key)
        for line in current_state_number.split('\n'):
            break
        match = re.search(r'\d+', line)
        if match:
            extracted_current_integer = int(match.group())
        else:
            assert False, 'no number in current state'

        previous_state = self.parent_node_of_node[key][current_state_number]
        previous_state_to_be_checked = self.get_prev_state_number(key)
        if previous_state != previous_state_to_be_checked:
            assert False, f'key is {key}, during backtrack, previous state marked and previous state by system are not the same'
        for line in previous_state.split('\n'):
            break
        match = re.search(r'\d+', line)
        if match:
            extracted_previous_integer = int(match.group())
        else:
            assert False
        return f"no solution, return to state {extracted_previous_integer} [that leads to state {extracted_current_integer}]"
    def revise_output_list(self, output_text):
        output_line_list = output_text.split("\n")
        is_tactic = False
        for idx_tactic, line in enumerate(output_line_list):
            if '_tactic_' in line:
                is_tactic = True
                break

        if is_tactic==False:
            #print('output, warning: no tactic')
            return ['no_tactic','no_tactic']

        if "::: " in output_line_list[idx_tactic]:
            output_line_list[idx_tactic] = output_line_list[idx_tactic][4:]
        entered_tactic_list = output_line_list[idx_tactic:idx_tactic+2]

        if len(entered_tactic_list) == 1:
            return ['no_tactic','no_tactic']
        return entered_tactic_list
    def check_if_failure_per_key(self, key):
        if len(self.tactic_list_tree[key]['state_0:']) == 0 and self.get_current_state_number(key) == 'state_0:':
            print('triggered failure')
            return True
        else:
            return False
    def check_path_length(self,key):
        current_state_label = self.get_current_state_number(key)
        previous_state_label = current_state_label
        theorem_object_length = 1
        #print(f'key is {key}, current tactic list tree is {self.tactic_list_tree[key]}')
        while True:
            #print("state chain label is: ", previous_state_label)
            if previous_state_label == 'state_0:':
                break
            previous_state_label = self.parent_node_of_node[key][current_state_label]
            theorem_object_length += 1
            current_state_label = previous_state_label
        #print(f'prompt tactic list length is {len(self.prompts_tactic_state_list[key])}')
        #print(f'theorem_object_length is {theorem_object_length}')
        if theorem_object_length != len(self.prompts_tactic_state_list[key]):
            assert True==False, "path_length not equal to each other"
    def check_if_program_finished(self):
        stop_signal = True
        for key in self.test_theorem_list:
            if self.root[key] == 'open':
                stop_signal = False
            else:
                pass
        return stop_signal
    def revise_prompt(self, prompts_tactic_state_list_per_key):
        pattern = r'state_\d+:'
        matches = re.findall(pattern, prompts_tactic_state_list_per_key)
        state_order = {}
        order = 0
        for match in matches:
            if match not in state_order:
                state_order[match] = order
                order += 1

        for state, ord in state_order.items():
            prompts_tactic_state_list_per_key = prompts_tactic_state_list_per_key.replace(state, f'state_{ord}:')

        last_state_id = None
        output_prompt = []
        for line in prompts_tactic_state_list_per_key.split('\n'):
            if re.search('state_\d+:', line):
                last_state_id = line[6:-1]
            elif re.search('state_\d+_tactic_', line):
                line = f'state_{last_state_id}_tactic_0:'
            output_prompt.append(line)


        '''for idx, item in enumerate(prompts_tactic_state_list_per_key):
            temp_string = re.sub(r'state_(\d+):',f'state_{idx}:', item)
            prompts_tactic_state_list_per_key[idx] = re.sub(r'state_(\d+)_tactic_(\d+):',f'state_{idx-1}_tactic_0:', temp_string)'''
        return "\n".join(output_prompt)
    def status_report(self):
        counter_in_process = 0
        counter_success = 0
        counter_failed = 0
        counter_too_long = 0
        counter_failed_with_error = 0
        for key in self.test_theorem_list:
            if self.root[key] == 'open':
                counter_in_process += 1
            if self.root[key] == 'success':
                counter_success += 1
            if self.root[key] == 'failed':
                counter_failed += 1
            if self.root[key] == 'failed, too long':
                counter_too_long += 1
            if self.root[key] == 'failed with error':
                counter_failed_with_error += 1
        self.counter_success = counter_success
        self.counter_failed = counter_failed
        self.counter_failed_with_error = counter_failed_with_error
        self.counter_too_long = counter_too_long
        self.counter_in_process = counter_in_process
        if counter_success + counter_failed + counter_too_long + counter_in_process + counter_failed_with_error != len(test_theorem_list):
            assert False, 'success, failed, too long, in process, failed with error add up not equal to total number'
        print(f'saved_file_path is {self.saved_file_path}')
        print(f'total number of theorem is {len(self.test_theorem_list)}')
        print(f'proof success number is {self.counter_success}')
        print(f'proof failed number is {self.counter_failed}')
        print(f'proof failed with error number is {self.counter_failed_with_error}')
        print(f'proof too long number is {self.counter_too_long}')
        print(f'proof in process number is {self.counter_in_process}')

        count_lean_single_backtrack = 0
        count_lean_multiple_backtrack = 0
        count_lean_tactic_success = 0
        for key in test_theorem_list:
            count_lean_single_backtrack += self.count_lean_dict[key]['count_lean_single_backtrack']
            count_lean_multiple_backtrack += self.count_lean_dict[key]['count_lean_multiple_backtrack']
            count_lean_tactic_success += self.count_lean_dict[key]['count_lean_tactic_success']

        print(f'total lean count tactic success is {count_lean_tactic_success}')
        print(f'total lean count single backtrack is {count_lean_single_backtrack}')
        print(f'total lean count multiple backtrack is {count_lean_multiple_backtrack}')
    def collect_inference_result(self, key_to_be_infered, outputs):
        for idx, output_list in tqdm(enumerate(outputs), total=len(outputs), desc=f"Processing LLM output for Round {self.round}"):
            assinged_output_list_per_key = []
            for i in range(0, self.num_sampled_tactics):
                output_tactic = self.revise_output_list(output_list.outputs[i].text)
                if output_tactic[0] == 'no_tactic' or output_tactic[1] == 'no_tactic':
                    pass
                else:
                    assinged_output_list_per_key.append(output_tactic)
                #print(f'key is {key_to_be_infered[idx]}, output_tactic is {output_tactic}')
            #print()
            seen = set()
            unique_assigned_output_list_per_key = []
            for inner_list in assinged_output_list_per_key:
                inner_tuple = tuple(inner_list)
                if inner_tuple not in seen:
                    seen.add(inner_tuple)
                    unique_assigned_output_list_per_key.append(inner_list)
            #print(
            #    f'key is {key_to_be_infered[idx]}, state_number to be assigned new tactic list is {self.cstate_round[key_to_be_infered[idx]]}')
            #print(f'key is {key_to_be_infered[idx]}, Assigned tactic list is {unique_assigned_output_list_per_key}')
            self.tactic_list_tree[key_to_be_infered[idx]][self.cstate_round[key_to_be_infered[idx]]] = unique_assigned_output_list_per_key

    def current_state_obtained_list(self, key):
        if self.root[key] == 'open':
            self.cstate_round[key] = self.get_current_state_number(key)
        cstate = self.cstate_round[key]
        pickle.dump(cstate, open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/current_state_{key}_{self.round}_{self.experiment_id}.pkl','wb'))
    def search(self):
        tokenizer = llm.get_tokenizer()
        while True:
            self.round += 1
            print(f'Round {self.round}------')
            if self.check_if_program_finished() or self.round > 65:
                print('confirmed test theorem finished. exit.')
                self.status_report()
                break

            self.key_to_be_infered = []
            self.key_not_finished = []
            self.prompts_entered = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                list(tqdm(executor.map(self.current_state_obtained_list, self.test_theorem_list), total=len(self.test_theorem_list), desc=f"get current state for Round {self.round}"))

            key_prompt_length_list = []
            for key in tqdm(self.test_theorem_list, total=len(self.test_theorem_list), desc=f"check state for theorems for round {self.round}" ):
                if self.root[key] == 'open':
                    self.round_count[key] = self.round
                    self.key_not_finished.append(key)
                    self.cstate_round[key] = pickle.load(open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/current_state_{key}_{self.round}_{self.experiment_id}.pkl','rb'))
                    if self.tactic_list_tree[key][self.cstate_round[key]] == None:
                        prompt_per_key = self.revise_prompt('\n'.join(self.prompts_tactic_state_list[key]))
                        key_prompt_length_list.append(len(prompt_per_key.split()))
                        tokenized_prompt_per_key = tokenizer.encode(prompt_per_key)
                        if len(prompt_per_key.split()) < 1500 and len(tokenized_prompt_per_key) < 4000: # used to be 1500
                            self.key_to_be_infered.append(key)
                            self.prompts_entered.append(prompt_per_key)
                        else:
                            self.root[key] = 'failed, too long'
                            self.key_not_finished.remove(key)
            print(f'key open need inference before check length, length list is {key_prompt_length_list}')

            print(f'key to be infered is {self.key_to_be_infered}')



    
            sampling_params = SamplingParams(n=self.num_sampled_tactics, temperature=self.temperature, top_p=1,
                                             max_tokens=200)  # temperature is 1.2 at beginning
            outputs = llm.generate(self.prompts_entered, sampling_params)


            print('now we collect the inference')
            self.collect_inference_result(self.key_to_be_infered, outputs)
            print('inference collected')

            print(f'enter concurrent process with max_workers as {self.max_workers}')

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                list(tqdm(executor.map(self.search_per_key_per_step, self.key_not_finished), total=len(self.key_not_finished), desc=f"Lean_verifying for Round {self.round}"))

            '''for key in self.key_not_finished:
                self.search_per_key_per_step(key)'''


            for key in tqdm(self.key_not_finished, total=len(self.key_not_finished), desc=f"Retrieve results from concurrent processing for Round {self.round}"):
                report_key_file = pickle.load(open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file_{key}_{self.round}_{self.experiment_id}.pkl','rb'))
                self.count_lean_dict[key]['count_lean_single_backtrack'] += report_key_file['count_lean_single_backtrack']
                self.count_lean_dict[key]['count_lean_multiple_backtrack'] += report_key_file['count_lean_multiple_backtrack']
                self.count_lean_dict[key]['count_lean_tactic_success'] += report_key_file['count_lean_tactic_success']
                self.root[key] = report_key_file['key_status']
                self.tactic_list_tree[key] = report_key_file['tactic_list_tree']
                self.prompts_tactic_state_list[key] = report_key_file['prompts_tactic_state_list']
                self.theorem_object_dict[key] = report_key_file['theorem_object_dict']
                self.parent_node_of_node[key] = report_key_file['node_relation']
                self.whether_backtrack[key] = report_key_file['whether_backtrack']

            self.status_report()
            self.save_outcome()
    def search_per_key_per_step(self, key):
        #print(f'key is {key}, current tactic_list_tree for Round {self.round} is {self.tactic_list_tree[key]}')

        try:
            key_status = 'open'
            count_lean_tactic = 0
            count_lean_single_backtrack = 0
            count_lean_multiple_backtrack = 0
            whether_backtrack = False

            tactic_list_at_top_per_key = self.tactic_list_tree[key][self.get_current_state_number(key)]
            #print(f"key is {key}, we start to search")
            if tactic_list_at_top_per_key == None:
                assert False, f"tactic_list_at_top_per_key is None, key is {key}\ncurrent tactic list is {self.tactic_list_tree[key]}" \
                              f"\nwhether key in key_to_be_infered {key in self.key_to_be_infered}"

            if len(tactic_list_at_top_per_key) != 0:
                try:
                    #print(f'key is {key}, We apply tactic')
                    count_lean_tactic += 1
                    entered_tactic = self.revise_entered_tactic(tactic_list_at_top_per_key[0], key)
                    label_before_tactic = self.get_current_state_number(key)
                    #print(f'key is {key}, get proof before apply tactic is: {self.theorem_object_dict[key].get_current_lean_proof()}')

                    #print(f'key is {key}, current tactic_list_tree before apply tactic is {self.tactic_list_tree[key]}')
                    #print(f'key is {key}, entered_tactic is {entered_tactic}')
                    #print(f'key is {key}, current state before apply tactic is: {label_before_tactic}')


                    lean_output = self.theorem_object_dict[key].provide_tactic(entered_tactic[0], entered_tactic[1])
                    label_after_tactic = self.get_current_state_number(key)
                    #print(f'key is {key}, current state after apply tactic is: {label_after_tactic}')

                    if 'proof is complete' in lean_output[1]:
                        key_status = 'success'
                        print(f'key is {key}, search is success!')
                        print(f'key is {key}, the successful proof \n{self.theorem_object_dict[key].get_current_lean_proof()}')
                        #self.status_report()

                    self.parent_node_of_node[key][label_after_tactic] = label_before_tactic
                    self.prompts_tactic_state_list[key].append(
                        f"{entered_tactic[0]}\n{entered_tactic[1]}\n{lean_output[1]}")
                    self.tactic_list_tree[key][self.get_current_state_number(key)] = None
                    del self.tactic_list_tree[key][label_before_tactic][0]
                    #print(f'key is {key}, current tactic_list_tree after apply tactic is {self.tactic_list_tree[key]}')
                    self.check_path_length(key)
                except Exception as e:
                    #print('tactic error happen')
                    #print(f'key is {key}, we apply tactic {entered_tactic} and see error')
                    #print(e)
                    #print(f'key is {key}, current state number after error after apply tactic is: {self.get_current_state_number(key)}')
                    #print(f'key is {key}, get proof after error after apply tactic is: {self.theorem_object_dict[key].get_current_lean_proof()}')
                    #print(f'key is {key}, current tactic_list_tree after error after apply tactic is {self.tactic_list_tree[key]}')
                    #print(f'key is {key}, we now delete the tactic from tactic list tree, the label is {self.get_current_state_number(key)}')
                    del self.tactic_list_tree[key][self.get_current_state_number(key)][0]
                    #print(f"key is {key}, tactic from tactic list tree is deleted, current tactic_list_tree is {self.tactic_list_tree[key]}")
                    self.check_path_length(key)
                    if self.check_if_failure_per_key(key):
                        key_status = 'failed'
                        print(f"key is {key}, tactic error then search failed!")
                        #self.status_report()
            else:
                if self.check_if_failure_per_key(key):
                    key_status = 'failed'
                    print(f"key is {key}, backtrack to zero and no tactic to try, search failed!")
                    #self.status_report()

                print(f'key is {key}, backtrack phase activated')
                whether_backtrack = True
                count_lean_single_backtrack += 1
                while True:
                    tactic_list_at_intermediate_node = self.tactic_list_tree[key][self.get_current_state_number(key)]
                    # print(f'tactic_tree_list: {self.tactic_list_tree[key]}')
                    # print(f"current state number: {self.get_current_state_number(key)}")
                    if len(tactic_list_at_intermediate_node) != 0:
                        break
                    if self.check_if_failure_per_key(key):
                        key_status = 'failed'
                        print(f"key is {key}, backtrack to zero and no tactic to try, search failed!")
                        #self.status_report()
                        break
                    #print(f'key is {key}, before back track step check length')
                    self.check_path_length(key)
                    #print(f'key is {key}, current state before backtrack is {self.get_current_state_number(key)}')
                    count_lean_multiple_backtrack += 1
                    lean_output = self.theorem_object_dict[key].do_back_track(self.back_track_tactic(key))
                    #print(f'current state after backtrack is {self.get_current_state_number(key)}')
                    #print('before delete the last ele of prompts_tactic_state_list')
                    #print('prompts_tactic_state_list is:')
                    #print(self.prompts_tactic_state_list[key])
                    #print(f'key for remove prompt here is {key}')

                    del self.prompts_tactic_state_list[key][-1]

                    #print('after back track step check length')
                    self.check_path_length(key)
        except Exception as e:
            print(f"key is {key}, exception happend")
            print(e)
            key_status = 'failed with error'
            count_lean_tactic = 0
            count_lean_single_backtrack = 0
            count_lean_multiple_backtrack = 0

        finally:
            report_key_file = {}
            report_key_file['count_lean_multiple_backtrack'] = count_lean_multiple_backtrack + count_lean_tactic
            report_key_file['count_lean_single_backtrack'] = count_lean_single_backtrack + count_lean_tactic
            report_key_file['count_lean_tactic_success'] = count_lean_tactic
            report_key_file['key_status'] = key_status
            report_key_file['tactic_list_tree'] = self.tactic_list_tree[key]
            report_key_file['prompts_tactic_state_list'] = self.prompts_tactic_state_list[key]
            report_key_file['theorem_object_dict'] = self.theorem_object_dict[key]
            report_key_file['node_relation'] = self.parent_node_of_node[key]
            report_key_file['whether_backtrack'] = whether_backtrack
            pickle.dump(report_key_file, open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file_{key}_{self.round}_{self.experiment_id}.pkl','wb'))



    def save_outcome(self):
        counter_in_process = 0
        counter_success = 0
        counter_failed = 0
        counter_too_long = 0
        count_lean_single_backtrack = 0
        count_lean_multiple_backtrack = 0
        count_lean_tactic_success = 0
        counter_failed_with_error = 0

        proof_dict = {}
        for key in tqdm(self.test_theorem_list,total=len(self.test_theorem_list),desc='saving results'):
            count_lean_single_backtrack += self.count_lean_dict[key]['count_lean_single_backtrack']
            count_lean_multiple_backtrack += self.count_lean_dict[key]['count_lean_multiple_backtrack']
            count_lean_tactic_success += self.count_lean_dict[key]['count_lean_tactic_success']
            proof_dict[key] = self.theorem_object_dict[key].get_current_lean_proof()
            if self.root[key] == 'open':
                counter_in_process += 1
            if self.root[key] == 'success':
                counter_success += 1
            if self.root[key] == 'failed':
                counter_failed += 1
            if self.root[key] == 'failed, too long':
                counter_too_long += 1
            if self.root[key] == 'failed with error':
                counter_failed_with_error += 1
        self.counter_success = counter_success
        self.counter_failed = counter_failed
        self.counter_failed_with_error = counter_failed_with_error
        self.counter_too_long = counter_too_long
        if counter_success + counter_failed + counter_too_long + counter_failed_with_error + counter_in_process!= len(test_theorem_list):
            assert False, 'number of theorm not equal to success, failed or too long, or in process'
        outcome = {}
        outcome['stats'] = {}
        outcome['stats']['total_lean_count_single_backtrack'] = count_lean_single_backtrack
        outcome['stats']['total_lean_count_multiple_backtrack'] = count_lean_multiple_backtrack
        outcome['stats']['count_lean_tactic_success'] = count_lean_tactic_success
        outcome['stats']['num_success'] = self.counter_success
        outcome['stats']['num_failed'] = self.counter_failed
        outcome['stats']['num_failed_with_error'] = self.counter_failed_with_error
        outcome['stats']['num_too_long'] = self.counter_too_long
        outcome['stats']['num_sampled_tactics'] = self.num_sampled_tactics
        outcome['stats']['temperature'] = self.temperature
        outcome['key_final_state'] = self.root
        outcome['key_lean_count'] = self.count_lean_dict
        outcome['key_proof'] = proof_dict
        outcome['tactic_list_tree'] = self.tactic_list_tree
        outcome['round_count'] = self.round_count

        pickle.dump(outcome, open(self.saved_file_path, 'wb'))

if __name__ == '__main__':
    '''os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if os.environ.get("TOKENIZERS_PARALLELISM") == "true":
        print("TOKENIZERS_PARALLELISM is set to true")
    else:
        print("TOKENIZERS_PARALLELISM is not set to true")'''

    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument('checkpoint_path', type=str, help='checkpoint_path')
    parser.add_argument('number_of_gpu', type=int, help='number_of_gpu')
    parser.add_argument('test_data_path', type=str, help='test_data_path')
    parser.add_argument('saved_file_path', type=str, help='test_data_path')
    parser.add_argument('max_workers', type=str, help='test_data_path')
    parser.add_argument('num_sampled_tactics', type=str, help='test_data_path')
    parser.add_argument('temperature', type=str, help='test_data_path')
    parser.add_argument('num_test_theorem', type=str, help='test_data_path')

    args = parser.parse_args()
    checkpoint = args.checkpoint_path
    number_of_gpu = args.number_of_gpu
    test_data_path = args.test_data_path
    saved_file_path = args.saved_file_path
    max_workers = args.max_workers
    num_sampled_tactics = args.num_sampled_tactics
    temperature = args.temperature
    num_test_theorem = args.num_test_theorem
    swap_space = 100

    print(f'checkpoint is {checkpoint}')
    print(f'number_of_gpu is {number_of_gpu}')
    print(f'test_data_path is {test_data_path}')
    print(f'saved_file_path is {saved_file_path}')
    print(f'num_test_theorem is {num_test_theorem}')


    print(f'max_workers is {max_workers}')
    print(f'number_sampled_tactic is {num_sampled_tactics}')
    print(f'temperature is {temperature}')

    print(f'swap_space is {swap_space}')


    random.seed(42)
    with open(test_data_path, 'r') as f:
        test_theorem_list = json.load(f)


    test_theorem_list = test_theorem_list[:int(num_test_theorem)]
    #print(test_theorem_list)
    #test_theorem_list = [9573315344600956080853155758,]

    experiment_id = uuid.uuid4()  # Generates a random UUID.
    print(experiment_id)



    llm = LLM(model=checkpoint, tensor_parallel_size=number_of_gpu, swap_space=swap_space)
    evaluate_obj = DFS(num_sampled_tactics=int(num_sampled_tactics), temperature=float(temperature), test_theorem_list=test_theorem_list, max_workers=int(max_workers), saved_file_path=saved_file_path, experiment_id=experiment_id)
    evaluate_obj.search()
    print('Now we start saving')
    evaluate_obj.save_outcome()
    print('Now we finish saving. exit')
    #command = "rm DFS/temp/temp_file*"
    #result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)







