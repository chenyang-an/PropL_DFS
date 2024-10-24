from vllm import LLM, SamplingParams

llm = LLM(model="/data/shared/llama-hf/llama-2-7b-hf", tensor_parallel_size=1, swap_space=4)

tokenizer = llm.get_tokenizer()

test_1 = tokenizer.encode("what is 1+1")
print(test_1)


prompt_token_ids = tokenizer.encode(["what is 1+1",'what is 2+2'])

print(prompt_token_ids)



# Truncate prompt_token_ids
prompt_token_ids = [ex[-4096:] for ex in prompt_token_ids]
print(prompt_token_ids)
sampling_params = SamplingParams(n=1, temperature=0, top_p=1,
                                 max_tokens=200)
outputs = llm.generate(prompt_token_ids=prompt_token_ids)
for output in outputs:
    print(output.outputs[0].text)