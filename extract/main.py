# -*- coding: utf-8 -*-
from llm_tools import process_ask_gpt_in_parallel,parse_json_response,ask_gpt
import glob
import json
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
import os
import time
import concurrent.futures
from extract.CodeLATS.code_lats import use_lats
#TODO: natural language interface with docs probably necessary
############ SUMMARIZE DOCUMENTATION ################
SUMMARIZE_DOCUMENTATION_DOC_PROMPT_TEMPLATE="""
Below is the content of an article of documentation on the library {library_name}.
Can you please create a through and concise summary of this article.
Please format your answer in json format as follows: 
    {{"summary":summary of article}}

Here is the article: 
{article}

Begin!
"""

############ CHOOSE DOCS TO READ FOR FINDING HOW TO RUN/WHAT CAN BE RUN  ################
WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE="""
Below are the summaries and titles of articles in the documentation of the library {library_name}.
You need to learn what simulations you can run and how to run them using this library from the documentation.
{specific_request}
You will be able to read the full documentation next.
What are the next {n} articles you would like to read to find information on what simulations you can run and how to run them?
Please place the articles in order of to be read first to to be read last. Please make sure that your answer only has {n} entries.
Please format your answer in json format as follows: 
    {{"articles":[list of article numbers]}}

Here are the summaries: 
{summaries}

Begin!
"""
    ##### TELL ME HOW TO RUN THIS EXPERIMENT ###############

HOW_TO_RUN_PROMPT_TEMPLATE="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
If it does contain information on how to run a simulation, summarize the simulation to be run and how modifications can be made to the simulation.
Be sure to include the locations or names of files to be modified (if they are in the article), as well as specific parameter names (if they are in the article)
If it does not contain information on how to run a simulation, say so and summarize the article.
Please format your answer in json format as follows: 
    {{"summary":summary}}

Here is the article: 
{article}

Begin!
"""

HOW_TO_RUN_PROMPT_TEMPLATE_part_1="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
If the article contains information on how to run a simulation, find the names of all of the files which can be modified to modify the simulation.
Then find all the parameter names which can be modified, a description of each, how to modify them and possible options for the parameter.

Please format your answer in json format as follows: 
    
    {{"is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "modification_code_files":[list of file names which can be modified to modify the simulation including the directory name]}}
    
    Note: If there is no information on how to run the simulation then the key modification_code_files is not to be used.


Here is the article: 
{article}

Begin!
"""

HOW_TO_RUN_PROMPT_TEMPLATE_part_12="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
If the article contains information on how to run a simulation, find the names of ALL of the files which can be modified to modify the simulation.
Then find ALL the parameter names which can be modified, a description of each, how to modify them and possible options for the parameter.

Please format your answer in json format as follows: 
    
    {{"is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "modification_code_files":[list of file names which can be modified to modify the simulation including the directory name],
      ""editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format]}}
    
    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"how_to_edit":how to edit the parameter,"options":the options for the parameter,"code_file":code file where the parameter can be edited including directory name}}
    
    Note: If there is no information on how to run the simulation then the keys modification_code_files,editable_parameters is not to be used.


Here is the article: 
{article}

Begin!
"""

edit_code_2="""
Write a function which changes the {variable_print} values in the text below to user specified values - \n\n{text}"""

GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT="""
You will be given the contents of a code file called {code_file_name} which contains options for configuring a simulation and the contents of the documentation page on configuring that simulation from the library {library_name}.
Use this information to find ALL of the parameters in the given code file which modify the simulation. Find their names, a description of each, how to modify them and possible options for the parameter.


Please format your answer in json format as follows: 
    
    {{""editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format]}}
    
    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"how_to_edit":how to edit the parameter,"options":the options for the parameter,"code_file":code file where the parameter can be edited including directory name}}

MAKE SURE THAT ALL OF THE PARAMETERS WHICH CAN BE EDITED IN THE CODE FILE ARE INCLUDED IN YOUR ANSWER.

Here is the documentation: 
{article}

Here is the code file named {code_file_name}:
{code}    

Begin!
"""

extracted_file_names=['verification/tutorial_barotropic_gyre/code/SIZE.h',
 'verification/tutorial_barotropic_gyre/input/data',
 'verification/tutorial_barotropic_gyre/input/data.pkg',
 'verification/tutorial_barotropic_gyre/input/eedata',
 'verification/tutorial_barotropic_gyre/input/bathy.bin',
 'verification/tutorial_barotropic_gyre/input/windx_cosy.bin']

HOW_TO_RUN_PROMPT_TEMPLATE_part_3="""
Below is an article in the documentation of the library {library_name}.
First generate a summary of the article.
Then determine if the article contains information on how to run a simulation.
If the article contains information on how to run a simulation, determine how the model can be run.
Please format your answer in json format as follows: 
    
    {{"summary":summary of the article,
      "is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "run_instructions":[list of commands to be run to run the model, in order],
      "run_simulation_description":a natural language description of how to run the model}}
    
    Note: If there is no information on how to run the simulation then the keys run_instructions,run_simulation_description are not to be used.


Here is the article: 
{article}

Begin!
"""

HOW_TO_RUN_PROMPT_TEMPLATE_part_4="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
If the article contains information on how to run a simulation, find out what the simulation outputs including the information in the dictionary below.
Please format your answer in json format as follows: 
    
    {{"is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "output_file_names":[list of ALL output file names of the simulation including the directory name],
      "output_variable_names":[list of dictionaries for ALL output variables, see below for output variables dictionary format ]}}
    
    output variables dictionary format : {{"output_variable_name":name of output variable,"description":description of output variable,"output_file":output file where the output variable can be found including directory name}}
    
    Note: If there is no information on how to run the simulation then the keys output_file_names,output_variable_names are not to be used.


Here is the article: 
{article}

Begin!
"""

############ CHOOSE DOCS TO READ FOR FINDING CONFIG VARIABLES  ################
WHICH_DOC_TO_READ_CONFIG_PROMPT_TEMPLATE="""
Below are the summaries of articles in the documentation of the library {library_name}.
You need to extract configuration variables which can be edited in using this library from the documentation.
You will be able to read the full documentation next.
What are the next 10 articles you would like to read to find configuration variables?
Please place the articles in order of to be read first to to be read last.
Please format your answer in json format as follows: 
    {{"articles":[list of article numbers]}}

Here are the summaries: 
{summaries}

Begin!
"""


############ EXTRACT PARAMETERS FROM DOC  ################
EXTRACT_FROM_DOC_PROMPT_TEMPLATE="""
Below is the content of an article of documentation from the library {library_name}.
You need to extract information about parameters which can be changed while using {library_name}.
Please format your answer in json format as follows: 
    {{"parameters":[list of dictionaries about parameters]}}
    
Each dictionary about a configuration variable can have the following keys:
    {{"name": name of the parameter,
      "description":description of the parameter,
      "type": the type of the parameter, could be str,float, etc..,
      "file": the name of the file where you can edit this parameter,
      "default":default value of the parameter,
      "options":possible options for the parameter}}
    
The name key is mandatory, the rest of the keys are optional

Here is the article: 
{article}

Begin!
"""
get_config_simple_1="""Please get a list of all the configuration variables from this fortran config file. 
Please format your answer in json format - {{'configuration_variables':[list of variables names]}} - 
{config_text}"""

get_config_simple_2_with_config="""For each of these variables create a dictionary with the variable name,
 the type of the variable (str, int, etc..),
 the default value of the variable (the value it is currently set at),
 a description of the variable and a description of the options the variable could be (ex. positive integers from 1 to 10..) 
 Please format your answer in json format - {{'configuration_variable_details':[list of dicts, with one dict for each variable in format below]}}
 Variable dict format - {{'variable_name':the exact name of the variable as given,'type':type of the variable (str, int, etc..),'default':the default value of the variable (the value it is currently set at), 'description':a description of the variable,'options':a description of the options the variable could be (ex. positive integers from 1 to 10..)}}
 Here is the configuration file for context - {documentation} 
 
 Here are the variables - 
 {variables}"""
 

code_mod_prompt_4="""Write a function which changes the {variables} values in the text below to user specified values. 
Note that the text below will ALWAYS be EXACTLY the same. 
You can take advantage of this fact in your function. 
Here is the function signature and doc string - 
def answer(original_text,\n{variables}):
#takes user supplied inputs and original text exactly in the form given and outputs text with the supplied variable values changed

Here is the original text -  
{text}"""

def get_modification_functions(variables,code):
    
    def use_lats_parallel(variable_slice):
        return use_lats(code_mod_prompt_4.format(variables=',\n'.join(variable_slice), text=code),model='gpt-3.5-turbo-1106',tree_depth=3)#'gpt-3.5-turbo-1106''gpt-4-1106-preview'
    num_workers = 9  
    
    functions = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(use_lats_parallel, variables[i:i+10]) for i in range(0, len(variables), 10)]
    
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # Save the result in the list
                functions.append(result)
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")
    return functions

def use_generate_functions_to_modify_config(functions,variable_details,user_inputs,config_text):
    """
    

    Parameters
    ----------
    functions : list 
        list of functions in string format.
    variable_details : dict
        dict of variable names, types, maybe others...
    user_inputs : dict
        Dict with key value pairs of parameter names to change and values to change them to
    config_text : str
        config_text in string format. Must be the initial string.

    Returns
    -------
    string.

    """
    #change names to answer1,answer2, etc..
    #add invalid name catching or invalid type catching from what model extracted.. (variable details)
    #eval functions so they can be used.
    #go over functions passing config text between functions
    for i,func in enumerate(functions):
        valid_input_variables=func.split('def answer(')[1].split('):')[0].split(',')
        valid_input_variables=[v.strip() for v in valid_input_variables]
        input_subset={}
        try:
            for key in variable_details.keys():
                if key in valid_input_variables:
                    if key in user_inputs.keys():
                        print(key)
                        value=user_inputs[key]
                    else:
                        value=variable_details[key]['default']
                    if type(value)==str:
                        if variable_details[key]['type']=='int':
                            input_subset[key]=int(value)
                        elif variable_details[key]['type']=='float':
                            input_subset[key]=float(value)
                        elif variable_details[key]['type']=='bool':
                            input_subset[key]=value
                        else:
                            input_subset[key]=value
                    else:
                        input_subset[key]=value
    
            func=func.replace('def answer(',f'def answer{i}(')
            #remove extra code like examples, etc.. - 
            save=False
            lines=[]
            for line in func.split('\n'):
                if 'def' in line:
                    save=True
                if save:
                    lines.append(line)
                if 'return' in line:
                    save=False
            func='\n'.join(lines)
            
            exec(func,globals())
            dynamic_func = globals()[f'answer{i}']
            config_text=dynamic_func(config_text,**input_subset)
        except:
            continue
    return config_text
    

def integrated_pipeline(docs_directory='/media/hdd/Code/auto-config/MITgcm/doc',code_directory='/media/hdd/Code/auto-config/MITgcm',library_name='MITGCM'):
    doc_files=glob.glob(docs_directory+'/**',recursive=True)
    docs_in_string_format=[]
    valid_doc_files=[]
    for file_path in doc_files:
        if '.rst' in file_path or '.md' in file_path:
            with open(file_path, 'r') as file:
                file_content = file.read()
            docs_in_string_format.append(file_content)
            valid_doc_files.append(file_path)
            
    #############   STEP 1: get summaries of documentation articles ####################
    prompts={j: SUMMARIZE_DOCUMENTATION_DOC_PROMPT_TEMPLATE.format(library_name=library_name,article=doc) 
             for j,doc in enumerate(docs_in_string_format)}
    summarize_docs_time=time.time()
    responses=process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
    print('Took this much time to summarize the docs - ',time.time()-summarize_docs_time)
    sorted_keys = sorted(responses.keys())
    responses = {key: responses[key] for key in sorted_keys}
    all_summaries = ' '.join([f"Article {i} : {json.loads(responses[key])['summary']}\n\n" for i,key in enumerate(responses.keys())])
    
    #############   STEP 2: WHICH DOCS TO READ  ####################
    
    which_articles_prompt=WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE.format(library_name=library_name,summaries=all_summaries,n=20,specific_request='')
    choose_docs_time=time.time()
    out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to choose which docs to read - ',time.time()-choose_docs_time)
    run_articles_to_read=json.loads(out)['articles']
    
    #############   STEP 3: GET CONFIG PARAMS FROM DOCS/CODE  ####################
    
       ########   STEP 3a: GET CONFIG FILES FROM DOCS  #############

    example_run_doc_1=docs_in_string_format[12]
    part12_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_12.format(library_name=library_name,article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(part12_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config files from docs - ',time.time()-read_doc_time)
    part12_extract=json.loads(out)   
    modification_code_files=part12_extract['modification_code_files']
    
       ########   STEP 3b: GET CONFIG PARAMS FROM DOC/CODE PAIRS  #############
    # 2 examples
    example_run_code_1_file_1_main='MITgcm/verification/tutorial_barotropic_gyre/code/SIZE.h'
    example_run_code_1_file_2='MITgcm/verification/tutorial_barotropic_gyre/input/data'
    
    with open(example_run_code_1_file_1_main, 'r') as file:
        file_content = file.read()
    example_run_code_1_file_1_main=file_content
    with open(example_run_code_1_file_2, 'r') as file:
        file_content = file.read()
    example_run_code_1_file_2=file_content
    
    doc_code_prompt=GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT.format(library_name=library_name,code_file_name=extracted_file_names[0],article=example_run_doc_1,code=example_run_code_1_file_1_main) #example_run_code_1_file_1_main
    read_doc_time=time.time()
    out = ask_gpt(doc_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config params from doc/code pair - ',time.time()-read_doc_time)
    docs_code_extract_1=json.loads(out)
    
    doc_code_prompt=GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT.format(library_name=library_name,code_file_name=extracted_file_names[1],article=example_run_doc_1,code=example_run_code_1_file_2) #example_run_code_1_file_2
    read_doc_time=time.time()
    out = ask_gpt(doc_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config params from doc/code pair - ',time.time()-read_doc_time)
    docs_code_extract=json.loads(out)
    
    #############   STEP 4: GET OUTPUT VARIABLES FROM OUTPUT FILES/DOCS  ####################

    output_file_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_4.format(library_name=library_name,article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(output_file_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get output files from docs - ',time.time()-read_doc_time)
    output_file_extract=json.loads(out)
    output_files=output_file_extract['output_file_names']
    output_variables=output_file_extract['output_variable_names']
    
    
    #############   STEP 5: GET RUN COMMANDS  ####################
    
    run_commands_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_3.format(library_name=library_name,article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(run_commands_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get run commands from docs - ',time.time()-read_doc_time)
    run_commands_extract=json.loads(out)
    run_commands=run_commands_extract['run_instructions']
                
