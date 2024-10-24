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
#CUDA_VISIBLE_DEVICES = 0 python -m DSF.DSF_search
class DFS:
    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        num_sampled_tactics: int,
        test_theorem_list,
        saved_file_path,
    ) -> None:
        self.tac_gen = tac_gen
        self.test_theorem_list = test_theorem_list
        self.num_sampled_tactics = num_sampled_tactics
        self.count_lean_dict = {}
        self.counter_success = 0
        self.counter_failed = 0
        self.counter_in_process = 0
        self.counter_too_long = 0
        self.saved_file_path = saved_file_path
        self.tactic_list_tree = {}
        self.theorem_object_dict = {}
        self.prompts_tactic_state_list = {}
        self.root = {}
        self.parent_node_of_node = {}
        self.round = 0
        print(f"{len(self.test_theorem_list)} many theorem loaded")
        for key in self.test_theorem_list:
            sample_eval = SingleTheoremEval(5, int(key))
            self.theorem_object_dict[key] = sample_eval
            init_state = self.theorem_object_dict[key].get_initial_prompt()

            self.parent_node_of_node[key] = {}
            self.prompts_tactic_state_list[key] = [init_state]
            self.root[key] = 'open' #open, success or failed
            self.tactic_list_tree[key] = {}
            self.tactic_list_tree[key]["state_0:"] = None
            self.parent_node_of_node[key]["state_0:"] = None
            self.count_lean_dict[key] = 0


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
            assert True == False

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
            print('output, warning: no tactic')
            return ['no_tactic','no_tactic']

        if "::: " in output_line_list[idx_tactic]:
            output_line_list[idx_tactic] = output_line_list[idx_tactic][4:]
        entered_tactic_list = output_line_list[idx_tactic:idx_tactic+2]
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
        while True:
            print("state chain label is: ",previous_state_label)
            if previous_state_label == 'state_0:':
                break
            previous_state_label = self.parent_node_of_node[key][current_state_label]
            theorem_object_length += 1
            current_state_label = previous_state_label
        print(f'prompt tactic list length is {len(self.prompts_tactic_state_list[key])}')
        print(f'theorem_object_length is {theorem_object_length}')
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
        for key in self.test_theorem_list:
            if self.root[key] == 'open':
                counter_in_process += 1
            if self.root[key] == 'success':
                counter_success += 1
            if self.root[key] == 'failed':
                counter_failed += 1
            if self.root[key] == 'failed, too long':
                counter_too_long += 1
        self.counter_success = counter_success
        self.counter_failed = counter_failed
        self.counter_too_long = counter_too_long
        self.counter_in_process = counter_in_process

        print(f'total number of theorem is {len(self.test_theorem_list)}')
        print(f'proof success number is {self.counter_success}')
        print(f'proof failed number is {self.counter_failed}')
        print(f'proof too long number is {self.counter_too_long}')
        print(f'proof in process number is {self.counter_in_process}')

        count_lean = 0
        for key in test_theorem_list:
            count_lean += self.count_lean_dict[key]

        print(f'total lean count {count_lean}')

    def collect_inference_result(self, key_to_be_infered, outputs):
        for idx, output_list in enumerate(outputs):
            assinged_output_list_per_key = []
            for i in range(0, self.num_sampled_tactics):
                output_tactic = self.revise_output_list(output_list.outputs[i].text)
                assinged_output_list_per_key.append(output_tactic)
                print(f'key is {key_to_be_infered[idx]}, output_tactic is {output_tactic}')
            print()
            seen = set()
            unique_assigned_output_list_per_key = []
            for inner_list in assinged_output_list_per_key:
                inner_tuple = tuple(inner_list)
                if inner_tuple not in seen:
                    seen.add(inner_tuple)
                    unique_assigned_output_list_per_key.append(inner_list)
            print(
                f'key is {key_to_be_infered[idx]}, state_number to be assigned new tactic list is {self.get_current_state_number(key_to_be_infered[idx])}')
            print(f'key is {key_to_be_infered[idx]}, Assigned tactic list is {unique_assigned_output_list_per_key}')
            self.tactic_list_tree[key_to_be_infered[idx]][self.get_current_state_number(
                key_to_be_infered[idx])] = unique_assigned_output_list_per_key

    def search(self):
        while True:
            self.round += 1
            print(f'Round {self.round}------')
            if self.check_if_program_finished():
                print('confirmed test theorem finished. exit.')
                self.status_report()
                break
            key_to_be_infered = []
            key_not_finished = []
            prompts_entered = []

            for key in self.test_theorem_list:
                if self.root[key] == 'open':
                    key_not_finished.append(key)
                    if self.tactic_list_tree[key][self.get_current_state_number(key)] == None:
                        prompt_per_key = self.revise_prompt('\n'.join(self.prompts_tactic_state_list[key]))
                        if len(prompt_per_key.split()) < 1500:
                            key_to_be_infered.append(key)
                            prompts_entered.append(prompt_per_key)
                        else:
                            self.root[key] = 'failed, too long'

            print(f'key to be infered is {key_to_be_infered}')

            sampling_params = SamplingParams(n=self.num_sampled_tactics, temperature=0, top_p=1, max_tokens=200, use_beam_search=True)
            outputs = self.tac_gen.generate(prompts_entered, sampling_params)

            self.collect_inference_result(key_to_be_infered, outputs)

            for key in tqdm(key_not_finished):
                self.search_per_key_per_step(key)

                report_key_file = pickle.load(open('/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file.pkl','rb'))
                self.count_lean_dict[key] += report_key_file['count_lean']
                self.root[key] = report_key_file['key_status']
                self.tactic_list_tree[key] = report_key_file['tactic_list_tree']
                self.prompts_tactic_state_list[key] = report_key_file['prompts_tactic_state_list']
                self.theorem_object_dict[key] = report_key_file['theorem_object_dict']
                self.parent_node_of_node[key] = report_key_file['node_relation']

            self.status_report()





#print(f"key is {key}, ")
    def search_per_key_per_step(self, key):
        key_status = 'open'
        count_lean = 0
        tactic_list_at_top_per_key = self.tactic_list_tree[key][self.get_current_state_number(key)]
        print(f"key is {key}, we start to search")



        if len(tactic_list_at_top_per_key) != 0:
            try:
                print(f'key is {key}, We apply tactic')
                count_lean += 1
                entered_tactic = self.revise_entered_tactic(tactic_list_at_top_per_key[0], key)
                label_before_tactic = self.get_current_state_number(key)
                print(
                    f'key is {key}, get proof before apply tactic is: {self.theorem_object_dict[key].get_current_lean_proof()}')

                print(f'key is {key}, current tactic_list_tree before apply tactic is {self.tactic_list_tree[key]}')
                print(f'key is{key}, entered_tactic is {entered_tactic}')
                print(f'key is {key}, current state before apply tactic is: {label_before_tactic}')


                lean_output = self.theorem_object_dict[key].provide_tactic(entered_tactic[0], entered_tactic[1])
                label_after_tactic = self.get_current_state_number(key)
                print(f'key is {key}, current state after apply tactic is: {label_after_tactic}')

                if 'proof is complete' in lean_output[1]:
                    key_status = 'success'
                    print(f'key is {key}, search is success!')
                    print(f'key is {key}, the successful proof \n{self.theorem_object_dict[key].get_current_lean_proof()}')
                    self.status_report()
                self.parent_node_of_node[key][label_after_tactic] = label_before_tactic
                self.prompts_tactic_state_list[key].append(
                    f"{entered_tactic[0]}\n{entered_tactic[1]}\n{lean_output[1]}")
                self.tactic_list_tree[key][self.get_current_state_number(key)] = None
                del self.tactic_list_tree[key][label_before_tactic][0]
                print(f'key is {key}, current tactic_list_tree after apply tactic is {self.tactic_list_tree[key]}')
                self.check_path_length(key)
            except Exception as e:
                print('tactic error happen')

                print(f'key is {key}, we apply tactic {entered_tactic} and see error')
                print(e)
                print(f'key is {key}, current state number after error after apply tactic is: {self.get_current_state_number(key)}')
                print(f'key is {key}, get proof after error after apply tactic is: {self.theorem_object_dict[key].get_current_lean_proof()}')
                print(f'key is {key}, current tactic_list_tree after error after apply tactic is {self.tactic_list_tree[key]}')
                print(f'key is {key}, we now delete the tactic from tactic list tree, the label is {self.get_current_state_number(key)}')
                del self.tactic_list_tree[key][self.get_current_state_number(key)][0]
                print(f"key is {key}, tactic from tactic list tree is deleted, current tactic_list_tree is {self.tactic_list_tree[key]}")
                self.check_path_length(key)
                if self.check_if_failure_per_key(key):
                    key_status = 'failed'
                    self.status_report()
        else:
            print(f'key is {key}, backtrack phase activated')
            print(f'key is {key}, current state_tactic_list before backtrack is {self.tactic_list_tree[key]}')
            count_lean += 1
            while True:
                tactic_list_at_intermediate_node = self.tactic_list_tree[key][self.get_current_state_number(key)]
                # print(f'tactic_tree_list: {self.tactic_list_tree[key]}')
                # print(f"current state number: {self.get_current_state_number(key)}")
                if len(tactic_list_at_intermediate_node) != 0:
                    break
                if self.check_if_failure_per_key(key):
                    key_status = 'failed'
                    print(f"key is {key}, search failed!")
                    self.status_report()
                    break
                print(f'key is {key}, before back track step check length')
                self.check_path_length(key)
                print(f'key is {key},current state before backtrack is {self.get_current_state_number(key)}')

                lean_output = self.theorem_object_dict[key].do_back_track(self.back_track_tactic(key))
                print(f'current state after backtrack is {self.get_current_state_number(key)}')
                print('before delete the last ele of prompts_tactic_state_list')
                print('prompts_tactic_state_list is:')
                print(self.prompts_tactic_state_list[key])
                print(f'key for remove prompt here is {key}')

                del self.prompts_tactic_state_list[key][-1]

                print('after back track step check length')
                self.check_path_length(key)

        report_key_file = {}
        report_key_file['count_lean'] = count_lean
        report_key_file['key_status'] = key_status
        report_key_file['tactic_list_tree'] = self.tactic_list_tree[key]
        report_key_file['prompts_tactic_state_list'] = self.prompts_tactic_state_list[key]
        report_key_file['theorem_object_dict'] = self.theorem_object_dict[key]
        report_key_file['node_relation'] = self.parent_node_of_node[key]
        pickle.dump(report_key_file, open('/home/c5an/leandojo_project/atp_research/DFS/temp/temp_file.pkl','wb'))



#python -m DFS.DFS_search
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument('checkpoint_path', type=str, help='checkpoint_path')
    parser.add_argument('number_of_gpu', type=int, help='number_of_gpu')
    parser.add_argument('test_data_path', type=str, help='test_data_path')
    parser.add_argument('saved_file_path', type=str, help='test_data_path')
    args = parser.parse_args()
    checkpoint = args.checkpoint_path
    number_of_gpu = args.number_of_gpu
    test_data_path = args.test_data_path
    saved_file_path = args.saved_file_path
    print(checkpoint)
    print(number_of_gpu)
    print(test_data_path)
    print(saved_file_path)

    with open(test_data_path, 'r') as f:
        test_theorem_list = json.load(f)

    test_theorem_list = test_theorem_list[:2]
    #test_theorem_list = [346960918443446424220011675436]
    #print(test_theorem_list)

    llm = LLM(model=checkpoint, tensor_parallel_size=number_of_gpu)
    num_sampled_tactics = 5
    print('number of sampled tactics is', num_sampled_tactics)
    evaluate_obj = DFS(llm, num_sampled_tactics=num_sampled_tactics, test_theorem_list=test_theorem_list, saved_file_path=saved_file_path, )

    tactic_list = [[["state_0_tactic_0:", "intro h1"],['stata_0_tactic_0', 'apply Or.inl']], [["state_1_tactic_0:", "apply Or.inl"]], [["state_2_tactic_0:", "intro h2"]],
              [["state_3_tactic_0:", "let h3 := h2.left"]],[["state_4_tactic_0:","let h4 := h2.right"]],[["state_5_tactic_0:", "cases h3"]],
              [["state_6_tactic_0:","case inl h5 =>d"]], ]

    evaluate_obj.search()


