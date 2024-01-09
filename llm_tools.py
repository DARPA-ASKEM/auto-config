
import time
import json
from tqdm import tqdm
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from typing import List,Union
import tenacity
import tiktoken
import re
def check_and_trim_tokens(prompt, model):
    encoding = tiktoken.get_encoding("cl100k_base")
    max_length_dict = {
        'gpt-4': 8000,
        'gpt-3.5-turbo-1106': 4096,
        'gpt-4-32k-0314c': 32768,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-4-32k-0613': 32768,
        'gpt-4-1106-preview':128000, #max tokens out is 4096
        "text-embedding-ada-002":8192,
    }
    max_tokens = max_length_dict[model]
    tokens=encoding.encode(prompt)
    if len(tokens) > max_tokens:
        print(f"Trimming prompt from {len(tokens)} characters to fit within token limit.")
        return encoding.decode(tokens[:max_tokens-100]) #100 just to be sure
    return prompt

def retry_decorator(retry_count=5, delay_seconds=10):
    """A decorator for retrying a function call with a specified delay and retry count."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < retry_count - 1:
                        time.sleep(delay_seconds)
            raise Exception(f"All {retry_count} attempts failed")
        return wrapper
    return decorator

@retry_decorator(retry_count=3, delay_seconds=5)
def ask_gpt(prompt, model='gpt-3.5-turbo-1106',**kwargs):
    """Send a prompt to GPT and get a response.
    
    Example Usage:
    Ex1:
        prompt = '''
            You will recieve a list of summaries of documents and a string which defines a topic of interest
            i.e. [summary_0, summary_1, summary_2, ...] topic
            You will respond only with in json form like:
            {"relevant_documents": [0,2], "non_relevant_documents": [1]},

            
            Note: All or none of the documents might be relevant
            Note: All documents should be assigned to either relevant or non-relevant
            
        '''
        prompt += "The list of summaries is:" + str(summaries)
        prompt += "The topic of interest is:" + str(topic)
        out = ask_gpt(prompt, model='gpt-3.5-turbo-1106')   
        
    """
    checked_prompt = check_and_trim_tokens(prompt, model)
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": checked_prompt}
        ],
        **kwargs
    )
    return response.choices[0].message.content.strip()


def parse_json_response(response):
    try:
        # Find the position of the first '{' and the last '}'
        first_brace_index = response.find('{')
        last_brace_index = response.rfind('}')
        
        # Check if both '{' and '}' are found
        assert (first_brace_index != -1 and last_brace_index != -1)
        # Extract the substring between the first '{' and the last '}'
        result = '{'+response[first_brace_index + 1:last_brace_index]+'}'
        out=json.loads(result)
    except:
        try:
            # Using regular expression to find JSON data in the format ```json{...}```
            pattern = r'```json(.+?)```'
            matches = re.findall(pattern, response, re.DOTALL)
            
            # Assuming we want to parse the first match only
            json_data = matches[0].strip()
            out = json.loads(json_data)
            #out=json.loads(response.split('```json')[1].strip().replace('```','')) #json md format, parse and load it
        except:
            try:
                out=json.loads(response) #json in string
            except:
                out=response #already in json
    if type(out)==dict:
        return out
    else:
        print('unable to parse into json')
        print(out)
        out=None
        return out

def process_ask_gpt_in_parallel(prompts, prompt_names, max_workers=8, model='gpt-3.5-turbo-16k',**kwargs):
    """
    example usage:
    ex1:
        def make_summaries(docs, names):
            #prompts = {n: f"Please provide a detailed one sentence summary of the following document: {d}" for n, d in zip(names, docs)}
            prompts = {n: f"Please provide a detailed one paragraph summary of the following document: {d}" for n, d in zip(names, docs)}
            return  process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model='gpt-3.5-turbo-16k')
    ex2:
        prompts={j: prompt.format(document=doc,
                                  topic=labels[label]['title'],
                                  description=labels[label]['description'],
                                  positive_examples_print=positive_examples_print,
                                  negative_examples_print=negative_examples_print) 
                 for j,doc in enumerate(docs)}
        responses=process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model=model,max_workers=8)
    
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ask_gpt, prompt, model,**kwargs): name for prompt, name in zip(prompts, prompt_names)}
        # Setting up tqdm progress bar
        with tqdm(total=len(prompts), desc="Processing Prompts") as progress:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    print(f"Error processing prompt '{name}': {e}")
                progress.update(1)  # Update the progress for each completed task
    return results

def setup_agentlm():
    import os
    def find_file_recursive(folder, target_filename):
        for root, _, files in os.walk(folder):
            for file in files:
                if file == target_filename:
                    return os.path.join(root, file)
    
        return None
    from langchain.llms import CTransformers
    cache_dir="/media/hdd/Code/llm-in-a-box/offline_models"
    file_name=find_file_recursive(cache_dir+'/models--TheBloke--agentlm-7B-GGUF','agentlm-7b.Q5_K_M.gguf')
    llm = CTransformers(model=file_name,model_type='llama',config={'gpu_layers':40,'context_length':4096,'max_new_tokens':512,'temperature':0,'repetition_penalty':1.1})    
    return llm

def ask_agentlm(prompt,llm):
    prompt_template=f"""[INST] <<SYS>>
       You are a helpful, respectful and honest assistant.
       <</SYS>>
       {prompt} [/INST]"""
    out=llm(prompt_template)  
    return out

@tenacity.retry(
    wait=tenacity.wait_fixed(20),
    stop=tenacity.stop_never,  # Retry indefinitely
    retry=tenacity.retry_if_exception_type(Exception)  # Retry on any exception
)
def get_openai_embeddings(string:Union[List,str]):
    if type(string)==list:
        string=[check_and_trim_tokens(s,"text-embedding-ada-002") for s in string]
    else:
        string=check_and_trim_tokens(string,"text-embedding-ada-002")
    response = openai.embeddings.create(
    model="text-embedding-ada-002",
    input=string
    )
    embeddings=[d.embedding for d in response.data]
    return embeddings
