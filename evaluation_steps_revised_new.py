
from propositional_logic.random_gen.evaluation_access import *
from vllm import LLM, SamplingParams
import time
import json
import pickle
import os
import time
import torch
import re
import sys
from tqdm import tqdm
import argparse
from loguru import logger

#python -m leandojo_project.proplogic_serv8.Evaluation.evaluation_steps_revised_new
#python -m leandojo_project.proplogic_serv8.Evaluation.evaluation_steps_revised /home/c5an/train_scripts/train_scripts_prop_serv8/train_llema_PropL_20w_1000/trained_model/checkpoint-400 1 /home/c5an/leandojo_project/proplogic_serv8/train_val_test/data_20w_1000_max/test_1000_key_list_ge_1200_le_1500.json /home/c5an/test.json

class status_holder:
    def __init__(self):
        self.counter_proof_success = 0
        self.counter_proof_error = 0
        self.counter_proof_too_long = 0
        self.counter_in_process = 0
        self.counter_proof_step = 0
        self.max_word_count = 1500
        self.max_output_length = 200
        self.total_count_tactic_lean = 0

    def print_status(self):
        print(f"Proof completed theorems {self.counter_proof_success}")
        print(f"Proof errors theorems {self.counter_proof_error}")
        print(f"Proof too long theorems {self.counter_proof_too_long}")
        print(f"Proof in process theorems {self.counter_in_process}")
        print(f"Total count tactic lean is {self.total_count_tactic_lean}")

def extract_tactic_from_output(output_line_list, index_tactic):
    revised_string_list = output_line_list[index_tactic:index_tactic + 2]

    if "::: " in revised_string_list[0]:
        revised_string_list[0] = revised_string_list[0][4:]

    entered_tactic_list = [revised_string_list[0], revised_string_list[1]]

    return entered_tactic_list

def check_proof_status_for_each_prompt(status_holder, prompts, iterated_key_list_of_prompts, prompt_length_under_search, prompt_length
                                       ,prompts_entered, output_dict):
    key_list_of_prompts = list(prompts.keys())
    for key in key_list_of_prompts:
        string_list = prompts[key].split()
        word_count = len(string_list)
        prompt_length.append(word_count)

        if "proof is complete" in prompts[key]:
            status_holder.counter_proof_success += 1
        elif "proof ends early" in prompts[key]:
            status_holder.counter_proof_error += 1
        elif word_count >= status_holder.max_word_count:
            prompts[key] = prompts[key] + f"\nOver {status_holder.max_word_count}"
            status_holder.counter_proof_too_long += 1
            output_dict[key]['proof_is_too_long'] = True
        else:
            status_holder.counter_in_process += 1
            prompts_entered.append(prompts[key])
            prompt_length_under_search.append(word_count)
            iterated_key_list_of_prompts.append(key)

def revise_output_list(outputs):
    output_list = []
    for idx, output in enumerate(outputs):
        output_text = output.outputs[0].text
        #output_text = output
        output_line_list = output_text.split("\n")
        if len(output_line_list) >= 2:
            if ('tactic' in output_line_list[0]) or ('tactic' in output_line_list[1]):
                pass
            elif ('no solution' in output_line_list[0]) or ('no solution' in output_line_list[1]):
                pass
            else:
                print(f"Warning{output_text}")

        else:
            print("Only one or zero line, warning:", output_line_list)
            output_line_list.append('Weired output')
            output_line_list.append('Weired output')
            output_text = "\n".join(output_line_list)

        output_list.append(output_text)
    return output_list



if __name__ == '__main__':
    print('new')
    parser = argparse.ArgumentParser(description='Description of your program.')

    parser.add_argument('checkpoint_path', type=str, help='checkpoint_path')
    parser.add_argument('number_of_gpu', type=int, help='number_of_gpu')
    parser.add_argument('test_data_path', type=str, help='test_data_path')
    parser.add_argument('saved_file_path', type=str, help='test_data_path')


    args = parser.parse_args()

    checkpoint = args.checkpoint_path
    number_of_gpu = args.number_of_gpu
    saved_file_path = args.saved_file_path
    test_data_path = args.test_data_path




    print(f'checkpoint is {checkpoint}')
    print(f'number_of_gpu is {number_of_gpu}')
    print(f'test_data_path is {test_data_path}')
    print(f'saved_file_path is {saved_file_path}')

    with open(test_data_path, 'r') as f:
        test_theorem_list = json.load(f)

    print(f'number of test data is {len(test_theorem_list)}')



    print('We finished loading data, now we initialize.')

    status_holder = status_holder()

    llm = LLM(model=checkpoint, tensor_parallel_size=number_of_gpu, swap_space=100)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=status_holder.max_output_length)

    init_state_list = []

    output_dict = {}
    prompts = {}
    error_info = {}

    theorem_object_dict = {}
    for i in range(0, len(test_theorem_list)):
        theorem_id = test_theorem_list[i]
        sample_eval = SingleTheoremEval(5, int(theorem_id))
        theorem_object_dict[theorem_id] = sample_eval
        init_state = theorem_object_dict[theorem_id].get_initial_prompt()

        temp_dict = {}
        temp_dict[theorem_id] = theorem_id
        temp_dict['initial_state'] = init_state
        temp_dict['generated_solution'] = ''
        temp_dict['steps'] = 0
        temp_dict['error_info'] = ''
        temp_dict['error_output'] = ''
        temp_dict['error_context'] = ''
        temp_dict['proof'] = ''
        temp_dict['proof_is_too_long'] = False
        temp_dict['proof_is_success'] = False
        temp_dict['num_no_solution'] = 0
        temp_dict['tactic_state_list'] = [init_state]
        temp_dict['count_tactic_lean'] = 0

        output_dict[test_theorem_list[i]] = temp_dict
        prompts[theorem_id] = init_state

    print("We've finished initialization, now we enter while true")
    print(f"Length of prompts is {len(prompts.keys())}")

    while True:
        status_holder.counter_proof_success = 0
        status_holder.counter_proof_error = 0
        status_holder.counter_proof_too_long = 0
        status_holder.counter_in_process = 0
        status_holder.counter_proof_step += 1

        prompts_entered = []
        prompt_length = []
        prompt_length_under_search = []
        iterated_key_list_of_prompts = []

        check_proof_status_for_each_prompt(status_holder, prompts, iterated_key_list_of_prompts,
                                           prompt_length_under_search, prompt_length
                                           , prompts_entered, output_dict)

        if len(prompts_entered) == 0:
            break

        print(f'Prompt length for all the theorems is {prompt_length}')
        print(f'Prompt length for theorems under search is {prompt_length_under_search}')
        print(f"There are {len(prompts_entered)} many prompts entered")
        print("Now making inference")
        outputs = llm.generate(prompts_entered, sampling_params)

        print("Inference are made")

        output_list = revise_output_list(outputs)

        print(f"------New Round------No. {status_holder.counter_proof_step}")
        print(f"iterated_key_list_of_prompts length is {len(iterated_key_list_of_prompts)}\n"
              f"output_list length is {len(output_list)}")
        print(f'saved_file_path is {saved_file_path}')

        for idx, key in tqdm(enumerate(iterated_key_list_of_prompts), total=len(iterated_key_list_of_prompts), desc="Lean_verifying"):
            error_info = ''

            output_line_list = output_list[0].split("\n")
            output_is_tactic = False
            output_is_no_sol = False
            try:
                for idx, line in enumerate(output_line_list):
                    if 'tactic' in line:
                        output_is_tactic = True
                        break
                    if 'no solution' in line:
                        output_is_no_sol = True
                        break
                if output_is_tactic:
                    status_holder.total_count_tactic_lean += 1
                    output_dict[key]['count_tactic_lean'] += 1
                    entered_tactic_list = extract_tactic_from_output(output_line_list,idx)
                    entered_tactic = '\n'.join(entered_tactic_list)
                    output_dict[key]['tactic_state_list'].append((output_list[0], entered_tactic))
                    lean_output = theorem_object_dict[key].provide_tactic(entered_tactic_list[0], entered_tactic_list[1])
                    output_dict[key]['tactic_state_list'].append(lean_output[1])
                    prompts[key] = prompts[key] + "\n" + entered_tactic + "\n" + lean_output[1]

                    if "proof is complete" in lean_output[1]:
                        output_dict[key]['proof_is_success'] = True

                elif output_is_no_sol:
                    back_track_input = output_line_list[idx]
                    output_dict[key]['tactic_state_list'].append((output_list[0], back_track_input))
                    lean_output = theorem_object_dict[key].do_back_track(back_track_input)
                    output_dict[key]['tactic_state_list'].append(lean_output[1])
                    prompts[key] = prompts[key] + "\n" + back_track_input + "\n" + lean_output[1]

                else:
                    output_dict[key]['tactic_state_list'].append(output_list[0])
                    print(f"Warning! Output is {output_list[0]}")
                    raise Exception("Model generate something contain neither tactic nor backtrack")

            except Exception as e:
                print(f"Error step theorem id is {key}")
                print(f"Original output is {output_list[0]}")
                print(f"An error occurred when apply tactic:\n{output_dict[key]['tactic_state_list'][-1]}")
                prompts[key] = prompts[key] + "\n" + "proof ends early"
                output_dict[key]['error_info'] = e
                pickle.dump(output_dict, open(saved_file_path, 'wb'))
            finally:
                output_dict[key]['steps'] = status_holder.counter_proof_step
                del output_list[0]

        print(f"the length of the output list of this round is now {len(output_list)}\n\n")
        status_holder.print_status()
        print(f'Test data path is {test_data_path}')
        print(f'Checkpoint is {checkpoint}')
        print(f"Saved file path is {saved_file_path}")
        print(f"Max length of solution is {status_holder.max_word_count}")

    key_list_of_prompts = list(prompts.keys())
    for key in key_list_of_prompts:
        output_dict[key]['generated_solution'] = prompts[key]
        output_dict[key]['proof'] = theorem_object_dict[key].get_current_lean_proof()

    print(f'Test data path is {test_data_path}')
    print(f'Checkpoint is {checkpoint}')
    print(f"Saved file path is {saved_file_path}")
    print(f"Max length of solution is {status_holder.max_word_count}")

    status_holder.print_status()

    temp_dict_1 = {}
    temp_dict_1['proof_success'] = status_holder.counter_proof_success
    temp_dict_1['total_theorem'] = len(key_list_of_prompts)
    temp_dict_1['proof_error'] = status_holder.counter_proof_error
    temp_dict_1[f'over_{status_holder.max_word_count}'] = status_holder.counter_proof_too_long
    temp_dict_1['model_checkpoint'] = checkpoint
    temp_dict_1['test_data_path'] = test_data_path
    temp_dict_1['total_count_tactic_lean'] = status_holder.total_count_tactic_lean

    output_dict['stats'] = temp_dict_1

    pickle.dump(output_dict, open(saved_file_path,'wb'))


