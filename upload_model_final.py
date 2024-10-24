from transformers import AutoModelForCausalLM, AutoTokenizer


import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Path to your local checkpoint directory
checkpoint_path = "/home/c5an/train_scripts/train_scripts_prop_serv8/v3_rand_0_4/checkpoint-3300"

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path)

# Load the model
model = LlamaForCausalLM.from_pretrained(checkpoint_path)
model.push_to_hub("KomeijiForce/llama-2-7b-propositional-logic-prover", token="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")
tokenizer.push_to_hub("KomeijiForce/llama-2-7b-propositional-logic-prover", token="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")

print('model loaded')