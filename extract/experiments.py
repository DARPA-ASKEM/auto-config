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
import re
from CodeLATS.code_lats import use_lats
#from langchain_community.document_loaders import UnstructuredMarkdownLoader
#from langchain_community.document_loaders import UnstructuredRSTLoader
#from langchain.text_splitter import MarkdownHeaderTextSplitter

#TODO: Find other libraries with different formats to find edge cases and see what is generic and what is not.. graphcast, others..
#TODO: implement stopping mechanism, when do we stop checking more documentation articles - after we stop getting configs? maybe we just ask gpt-3.5 if each one contains something relevant... Maybe we use GPT-4 like an agent to assess if there is more to look at..
#TODO: write into prompts distinguishing different simulations from each other.. One article/code file may have multiple sims in it..
#TODO: add validation (unit test gen, try changing configs, etc..?)
#TODO: add saving to avoid re-execution
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

#85 docs
docs_directory='/media/hdd/Code/auto-config/MITgcm/doc'
doc_files=glob.glob(docs_directory+'/**',recursive=True)
docs_in_string_format=[]
valid_doc_files=[]
for file_path in doc_files:
    if '.rst' in file_path or '.md' in file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
        docs_in_string_format.append(file_content)
        valid_doc_files.append(file_path)
#get documentation articles (.md,.rst)

def summarize_many_docs_example():
    #get summaries of documentation articles
    prompts={j: SUMMARIZE_DOCUMENTATION_DOC_PROMPT_TEMPLATE.format(library_name='MITGCM',article=doc) 
             for j,doc in enumerate(docs_in_string_format)}
    #'gpt-3.5-turbo-1106' 'gpt-4-1106-preview'
    responses=process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
    #TODO: handle issues of lots of stuff being cut off with gpt-3.5
    sorted_keys = sorted(responses.keys())
    responses = {key: responses[key] for key in sorted_keys}
    all_summaries = ' '.join([f"Article {i} : {json.loads(responses[key])['summary']}\n\n" for i,key in enumerate(responses.keys())])
    return all_summaries

#all summaries are 42964 length total..

############ CHOOSE DOCS TO READ FOR FINDING HOW TO RUN/WHAT CAN BE RUN  ################
WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE="""
Below are the summaries of articles in the documentation of the library {library_name}.
You need to learn what simulations you can run and how to run them using this library from the documentation.
You will be able to read the full documentation next.
What are the next {n} articles you would like to read to find information on what simulations you can run and how to run them?
Please place the articles in order of to be read first to to be read last.
Please format your answer in json format as follows: 
    {{"articles":[list of article numbers]}}

Here are the summaries: 
{summaries}

Begin!
"""

def example_which_docs_to_read(all_summaries):
    which_articles_prompt=WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE.format(library_name='MITGCM',summaries=all_summaries,n=20)
    out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    run_articles_to_read=json.loads(out)['articles']
    return run_articles_to_read
    
    #TODO: Need to account for the case where this information is spread over multiple files and we need to carry a dict around and update it as we go
    ##### TELL ME HOW TO RUN THIS EXPERIMENT ###############
    
example_run_code_1_file_1='/media/hdd/Code/auto-config/MITgcm/verification/tutorial_barotropic_gyre/code/SIZE.h'
example_run_code_1_file_2='/media/hdd/Code/auto-config/MITgcm/verification/tutorial_barotropic_gyre/input/data'
example_run_doc_1=docs_in_string_format[12]
example_run_doc_2=docs_in_string_format[2]
with open(example_run_code_1_file_1, 'r') as file:
    file_content = file.read()
example_run_code_1_file_1=file_content
with open(example_run_code_1_file_2, 'r') as file:
    file_content = file.read()
example_run_code_1_file_2=file_content

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
def extract_run_command_example(example_run_doc_1,example_run_doc_2):
    run_summary_1_prompt=HOW_TO_RUN_PROMPT_TEMPLATE.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(run_summary_1_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    run_summary_1=json.loads(out)['summary']
    
    run_summary_2_prompt=HOW_TO_RUN_PROMPT_TEMPLATE.format(library_name='MITGCM',article=example_run_doc_2)
    out = ask_gpt(run_summary_2_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    run_summary_2=json.loads(out)['summary']
    return run_summary_1,run_summary_2

#TODO: change prompt to handle multiple simulations in one file or otherwise break it down.

HOW_TO_RUN_PROMPT_TEMPLATE_2="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
Then summarize the article.
Then if the article contains information on how to run a simulation, find the names of all of the files which can be modified to modify the simulation.
Then find all the parameter names which can be modified, a description of each, how to modify them and possible options for the parameter.
Then if contained in the article, find how the model can be run.
Please format your answer in json format as follows: 
    
    {{"is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "summary":summary of the article,
      "modification_code_files":[list of file names which can be modified to modify the simulation including the directory name],
      "editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format],
      "run_instructions":[list of commands to be run to run the model, in order],
      "run_simulation_description":a natural language description of how to run the model,
      "output_file_names":[list of output file names of the simulation including the directory name],
      "output_variable_names":[list of dictionaries on output variable names, see below for output variables dictionary format ]}}
    
    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"how_to_edit":how to edit the parameter,"options":the options for the parameter,"code_file":code file where the parameter can be edited including directory name}}
    output variables dictionary format : {{"output_variable_name":name of output variable,"description":description of output variable,"output_file":output file where the output variable can be found including directory name}}
    Note: The keys modification_code_files, editable_parameters, run_instructions and run_simulation_description are only to be used when thearticle contains information on how to run a simulation.
    Note: If there is no information on how to run the simulation then keys run_instructions,run_simulation_description are not to be used.
    Note: If there is no information on the output file then keys output_file_names and output_variable_names are not to be used.


Here is the article: 
{article}

Begin!
"""
def how_to_run_template_2_example(example_run_doc_1):
    run_summary_1_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_2.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(run_summary_1_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    run_summary_1_extract=json.loads(out)
    return run_summary_1_extract

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

HOW_TO_RUN_PROMPT_TEMPLATE_part_2="""
Below is an article in the documentation of the library {library_name}.
First determine if the article contains information on how to run a simulation.
If the article contains information on how to run a simulation, find all the parameter names which can be modified, a description of each, how to modify them and possible options for the parameter.
Please format your answer in json format as follows: 
    
    {{"is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format]}}
    
    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"how_to_edit":how to edit the parameter,"options":the options for the parameter,"code_file":code file where the parameter can be edited including directory name}}
    Note: If there is no information on how to run the simulation then the key modification_code_files is not to be used.


Here is the article: 
{article}

Begin!
"""

#TODO: test does generating summary help performance?
GET_CONFIG_FILES_FROM_DOCS_PROMPT_TEMPLATE="""
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
#adding these two together gives better output files but doesn't get all the editable parameters.
def extract_info_example_2(example_run_doc_1):
    part12_prompt=GET_CONFIG_FILES_FROM_DOCS_PROMPT_TEMPLATE.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(part12_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    part12_extract=json.loads(out)
    return part12_extract

#TODO: more dynamically get the documentation that would be helpful in an agent style fashion.
GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT="""
You will be given the contents of a code file called {code_file_name} which contains options for configuring a simulation and the contents of the documentation page on configuring that simulation from the library {library_name}.
Use this information to find ALL of the parameters in the given code file which modify the simulation. Find their names, a description of each, how to modify them and possible options for the parameter.

Please format your answer in json format as follows: 
    
    {{""editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format]}}
    
    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"how_to_edit":how to edit the parameter,"options":the options for the parameter,"code_file":code file where the parameter can be edited including directory name}}


Here is the documentation: 
{article}

Here is the code file:
{code}    

Begin!
"""

get_config_simple_1="""Please get a list of all the configuration variables from this fortran config file. 
Please format your answer in json format - {{'configuration_variables':[list of variables names]}} - 
{config_text}"""
conf_result_1=ask_gpt(get_config_simple_1.format(config_text=example_run_code_1_file_1),'gpt-3.5-turbo-1106',response_format={"type": "json_object"}) #TODO: test 3.5 on this..
variables=json.loads(conf_result_1)['configuration_variables']

get_config_simple_2="""For each of these variables create a dictionary with the variable name,
 the type of the variable (str, int, etc..),
 a description of the variable and a description of the options the variable could be (ex. positive integers from 1 to 10..) - 
 {variables}""" #note this is done in the conv history of the previous one..
get_config_simple_2_with_docs="""For each of these variables create a dictionary with the variable name,
 the type of the variable (str, int, etc..),
 the default value of the variable (the value it is currently set at),
 a description of the variable and a description of the options the variable could be (ex. positive integers from 1 to 10..) 
 Please format your answer in json format - {{'configuration_variable_details':[list of dicts, with one dict for each variable in format below]}}
 Variable dict format - {{'variable_name':the exact name of the variable as given,'type':type of the variable (str, int, etc..),'default':the default value of the variable (the value it is currently set at), 'description':a description of the variable,'options':a description of the options the variable could be (ex. positive integers from 1 to 10..)}}
 Here is some related documentation for context - {documentation} 
 
 Here are the variables - 
 {variables}"""  #note this is done in the conv history of the previous one..
 
get_config_simple_2_with_config="""For each of these variables create a dictionary with the variable name,
 the type of the variable (str, int, etc..),
 the default value of the variable (the value it is currently set at),
 a description of the variable and a description of the options the variable could be (ex. positive integers from 1 to 10..) 
 Please format your answer in json format - {{'configuration_variable_details':[list of dicts, with one dict for each variable in format below]}}
 Variable dict format - {{'variable_name':the exact name of the variable as given,'type':type of the variable (str, int, etc..),'default':the default value of the variable (the value it is currently set at), 'description':a description of the variable,'options':a description of the options the variable could be (ex. positive integers from 1 to 10..)}}
 Here is the configuration file for context - {documentation} 
 
 Here are the variables - 
 {variables}"""
 
#get_details_prompts={i:get_config_simple_2_with_docs.format(variables='\n'.join(variables[i:i+10]),documentation=example_run_doc_1) for i in range(0,len(variables),10)}
get_details_prompts={i:get_config_simple_2_with_config.format(variables='\n'.join(variables[i:i+10]),documentation=example_run_code_1_file_1) for i in range(0,len(variables),10)}
detailed_responses=process_ask_gpt_in_parallel(get_details_prompts.values(), get_details_prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
detailed_responses={key:json.loads(detailed_responses[key]) for key in detailed_responses.keys()}
variable_details={}
for key in detailed_responses.keys():
    for d in detailed_responses[key]['configuration_variable_details']:
        name=d['variable_name']
        d.pop('variable_name')
        variable_details[name]=d

#TODO:try adding more to doc string..
#TODO: maybe give a list of the types it is allowed to be and other other stuff from details..
code_mod_prompt_4="""Write a function which changes the {variables} values in the text below to user specified values. 
Note that the text below will ALWAYS be EXACTLY the same. 
You can take advantage of this fact in your function. 
Here is the function signature and doc string - 
def answer(original_text,\n{variables}):
#takes user supplied inputs and original text exactly in the form given and outputs text with the supplied variable values changed

Here is the original text -  
{text}"""

import concurrent.futures
def get_modification_functions(variables,code):
    
    def parallel_function(variable_slice):
        # This is the function that will be executed in parallel
        # Replace `modify_code_function4` with the actual function call you need
        return use_lats(code_mod_prompt_4.format(variables=',\n'.join(variable_slice), text=example_run_code_1_file_1),model='gpt-3.5-turbo-1106',tree_depth=3)
    
    # Number of workers equals the number of threads you want to use
    # You can adjust this based on your system's capabilities
    num_workers = 9  
    
    functions = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Creating a list of futures
        futures = [executor.submit(parallel_function, variables[i:i+10]) for i in range(0, len(variables), 10)]
    
        # Retrieving and saving results as they are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # Save the result in the list
                functions.append(result)
                # Optionally, print the result here
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
        for key in variable_details.keys():
            if key in valid_input_variables:
                if key in user_inputs.keys():
                    print(key)
                    value=user_inputs[key]
                else:
                    value=variable_details[key]['default']
                if variable_details[key]['type']=='int':
                    input_subset[key]=int(value)
                elif variable_details[key]['type']=='float':
                    input_subset[key]=float(value)
                elif variable_details[key]['type']=='bool':
                    input_subset[key]=value
                else:
                    input_subset[key]=value

        func=func.replace('def answer(',f'def answer{i}(')
        exec(func,globals())
        dynamic_func = globals()[f'answer{i}']
        config_text=dynamic_func(config_text,**input_subset)
    return config_text
    


HOW_TO_COMPILE_PROMPT_TEMPLATE_part_3="""
Below is an article in the documentation of the library {library_name}.
First generate a summary of the article.
Then determine if the article contains information on how to compile a simulation.
If the article contains information on how to compile a simulation, determine how the model can be compiled.
Please format your answer in json format as follows: 

    {{"summary":summary of the article,
      "is_simulation_article":True/False (True is the article contains information on how to run a simulation, false otherwise),
      "compilation_instructions":[list of commands to be run to run the model, in order],
      "compilation_simulation_description":a natural language description of how to compile the model}}

    Note: If there is no information on how to compile the simulation then the keys compilation_instructions,compilation_simulation_description are not to be used.


Here is the article: 
{article}

Begin!
"""
library_name='WRF'
compilation_commands_prompt=HOW_TO_COMPILE_PROMPT_TEMPLATE_part_3.format(library_name=library_name,article=example_run_doc_1)
read_doc_time=time.time()
compile_out = ask_gpt(compilation_commands_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
print('Took this much time to get run commands from docs - ',time.time()-read_doc_time)


GET_CONFIG_PARAMS_MORE_EDITING_FROM_DOCS_AND_CODE_PROMPT="""
You will be given the contents of a code file called {code_file_name} which contains options for configuring a simulation and the contents of the documentation page on configuring that simulation from the library {library_name}.
The lines of the code file will be numbered.
Use this information to find ALL of the parameters in the given code file which modify the simulation. Find their names, a description of each, how to modify them and possible options for the parameter.

Please format your answer in json format as follows: 

    {{""editable_parameters":[list of dictionaries on parameters which can be modified, see below for editable parameters dictionary format]}}

    editable parameters dictionary format : {{"parameter_name":name of parameter,"description":description of parameter,"line_number":line number on which the parameter has a value assigned to it, "substring":the substring in the line which contains the parameter value, ex. in sx = 24 substring is 24,"type":type of the parameter}}


Here is the documentation: 
{article}

Here is the code file:
{code}    

Begin!
"""
#TODO: maybe give a list of the types it is allowed to be??


#example edit code output - 
# import re

# def edit_parameter_function(file_content, new_sNx):
#     # Check if new_sNx is a valid value (integer greater than 0)
#     if not isinstance(new_sNx, int) or new_sNx <= 0:
#         raise ValueError(f'Invalid sNx value: {new_sNx}. It must be an integer greater than 0.')

#     # Regular expression pattern to find and replace the sNx parameter value
#     sNx_pattern = re.compile(r'(sNx =)\n? +[0-9]+')

#     # Perform the substitution
#     new_file_content = sNx_pattern.sub(r'\1 ' + str(new_sNx), file_content)

#     return new_file_content
# new_file=edit_parameter_function(example_run_code_1_file_1,10)
# print(new_file)


GET_OUTPUT_PARAMS_FROM_DOCS_AND_FILE_PROMPT="""
You will be given the contents of a output file called {output_file_name} which contains some or all of the outputs of a simulation run and the contents of the documentation page on that simulation from the library {library_name}.
Use this information to find ALL of the output parameters in the given output file. Find their names and give a description of each.

Please format your answer in json format as follows: 
    
    {{""output_parameters":[list of dictionaries on output parameters, see below for output parameters dictionary format]}}
    
    output parameters dictionary format : {{"output_name":name of parameter,"description":description of parameter,"output_file":name of the output file which this parameter can be found in. Make sure this matches the output file name above, otherwise do not include the parameter}}


Here is the documentation: 
{article}

Here is the output file:
{output_file}    

Begin!
"""
#TODO: some way to do logging/human checking of process? Maybe save docs it looks at and highlight stuff it extracts, then we can see if it missed some?? streamlit?

extracted_file_names=['verification/tutorial_barotropic_gyre/code/SIZE.h',
 'verification/tutorial_barotropic_gyre/input/data',
 'verification/tutorial_barotropic_gyre/input/data.pkg',
 'verification/tutorial_barotropic_gyre/input/eedata',
 'verification/tutorial_barotropic_gyre/input/bathy.bin',
 'verification/tutorial_barotropic_gyre/input/windx_cosy.bin']

def get_info_from_doc_pairs_example(example_run_doc_1,example_run_code_1_file_2):
    doc_code_prompt=GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT.format(library_name='MITGCM',code_file_name=extracted_file_names[1],article=example_run_doc_1,code=example_run_code_1_file_2) #example_run_code_1_file_2
    out = ask_gpt(doc_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    docs_code_extract=json.loads(out)
    return docs_code_extract

EDIT_CODE_GIVEN_PARAM_PROMPT="""
You will be given the contents of a code file called {code_file_name} which contains the parameter {option} (among other parameters) for configuring a simulation from the library {library_name}.
Please write a short python function which can take the string content of the file below and change the parameter to a valid value.
Please include a detailed doc string and include types for the inputs.
Please also write a test for the function

Please format your answer in json format as follows: 

    {{""python code":python  markdown containing code with edit_parameter_function,
      "test":python markdown containing code including test}}

Here is the code: 

{code}
 
"""
#DONE: try LATS - https://github.com/andyz245/LanguageAgentTreeSearch/tree/main?tab=readme-ov-file https://huggingface.co/spaces/AIatUIUC/CodeLATS 
#TODO: maybe RLTF (rl from unit test feedback) - https://github.com/Zyq-scut/RLTF
edit_code_2="""
Write a function which changes the {variable_print} values in the text below to user specified values - \n\n{text}"""

edit_code_3="""Write a function which changes the {variable_print} values in the text below to user specified values. Note that the text below will ALWAYS be EXACTLY the same. You can take advantage of this fact in your function. - \n\n{text}"""

EDIT_CODE_GIVEN_PARAM_PROMPT="""
You will be given the contents of a code file called {code_file_name} which contains the parameter {option} (among other parameters) for configuring a simulation from the library {library_name} which is written in fortran.
Please write a short python function named edit_parameters which can take the string content of the file below and change the parameter to a valid value.
Please include a detailed doc string and include types for the inputs.
Please format your answer in json format as follows: 

    {{""python code":str of python code containing only the code for edit_parameters}}

Here is the code: 

{code}

"""

try2="""Write a function which changes the any of editable parameters in the text below to user specified values. Include a description of each variable in the doc string - 

 &time_control
 run_days                            = 5,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 0001, 0001, 0001,
 start_month                         = 01,   01,   01,
 start_day                           = 01,   01,   01,
 start_hour                          = 00,   00,   00,
 start_minute                        = 00,   00,   00,
 start_second                        = 00,   00,   00,
 end_year                            = 0001, 0001, 0001,
 end_month                           = 01,   01,   01,
 end_day                             = 05,   05,   05,
 end_hour                            = 00,   00,   00,
 end_minute                          = 00,   00,   00,
 end_second                          = 00,   00,   00,
 history_interval                    = 360,  360,  360,
 frames_per_outfile                  = 1000, 1000, 1000,
 restart                             = .false.,
 restart_interval                    = 3600,
 io_form_history                     = 2
 io_form_restart                     = 2
 io_form_input                       = 2
 io_form_boundary                    = 2
 /

 &domains
 time_step                           = 600,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 s_we                                = 1,     1,     1,
 e_we                                = 41,    41,    41,
 s_sn                                = 1,     1,     1,
 e_sn                                = 81,    81,    81,
 s_vert                              = 1,     1,     1,
 e_vert                              = 65,    65,    65,
 dx                                  = 100000,
 dy                                  = 100000,
 ztop                                = 16000, 16000, 16000,
 grid_id                             = 1,     2,     3,
 parent_id                           = 0,     1,     2,
 i_parent_start                      = 0,     17,    17,
 j_parent_start                      = 0,     33,    33,
 parent_grid_ratio                   = 1,     5,     5,
 parent_time_step_ratio              = 1,     5,     5,
 feedback                            = 1,
 smooth_option                       = 0
 /

 &physics
 mp_physics                          = 0,     0,     0,
 ra_lw_physics                       = 0,     0,     0,
 ra_sw_physics                       = 0,     0,     0,
 radt                                = 30,    30,    30,
 sf_sfclay_physics                   = 0,     0,     0,
 sf_surface_physics                  = 0,     0,     0,
 bl_pbl_physics                      = 0,     0,     0,
 bldt                                = 0,     0,     0,
 cu_physics                          = 0,     0,     0,
 cudt                                = 5,     5,     5,
 /

 &fdda
 /

 &dynamics
 hybrid_opt                          = 0, 
 rk_ord                              = 3,
 diff_opt                            = 1,      1,      1,
 km_opt                              = 1,      1,      1,
 damp_opt                            = 0,
 zdamp                               = 4000.,  4000.,  4000.,
 dampcoef                            = 0.01,   0.01,   0.01
 khdif                               = 0,      0,      0,
 kvdif                               = 0,      0,      0,
 smdiv                               = 0.1,    0.1,    0.1,
 emdiv                               = 0.01,   0.01,   0.01,
 epssm                               = 0.1,    0.1,    0.1
 time_step_sound                     = 4,      4,      4,
 h_mom_adv_order                     = 5,      5,      5,
 v_mom_adv_order                     = 3,      3,      3,
 h_sca_adv_order                     = 5,      5,      5,
 v_sca_adv_order                     = 3,      3,      3,
 non_hydrostatic                     = .true., .true., .true.,
 /

 &bdy_control
 periodic_x                          = .true., 
 symmetric_xs                        = .false.,
 symmetric_xe                        = .false.,
 open_xs                             = .false.,
 open_xe                             = .false.,
 periodic_y                          = .false.,
 symmetric_ys                        = .true., 
 symmetric_ye                        = .true., 
 open_ys                             = .false.,
 open_ye                             = .false.,
 /

 &grib2
 /

 &namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
 /

 &ideal
 ideal_case = 7
 /"""
 
def code_edit_experiments():
    #TODO: ask for example usage or unittests, ask for certain things in doc string, etc.. etc..
    edit_prompt=EDIT_CODE_GIVEN_PARAM_PROMPT.format(library_name='MITGCM',code_file_name=extracted_file_names[0],article=example_run_doc_1,code=example_run_code_1_file_1,option='sNx')
    out = ask_gpt(edit_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    edit_code=json.loads(out)
    print(edit_code['python code'])
    print(edit_code['test'])
    import openai
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": edit_prompt}
    ]
    response = openai.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=messages,
        response_format={"type": "json_object"}
    )
    message=response.choices[0].message
    out = response.choices[0].message.content.strip()
    code=json.loads(out)['python code']
    test=json.loads(out)['test']
    parsed_code=json.loads(out)['python code'].split("```")[1].replace("python",'').strip()
    parsed_test=json.loads(out)['test'].split("```")[1].replace("python",'').strip()
    print(parsed_code)
    print(parsed_test)
    
    import re
    function_pattern = r'(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\):\s*[\s\S]*?)(?=(?:\s*def|$))'
    function_matches = re.finditer(function_pattern, parsed_code)
    functions=[]
    for i,match in enumerate(function_matches):
        function_code = match.group(1).strip()
        functions.append(function_code)
    functions.append(function_code)
    
    messages.append(message)
    messages.append({"role": "user", "content": 'I tried running the test and your code and got an assertion error, fix the code.'})
    

#TODO: write code to verify and debug
DEBUG_EDIT_CODE="""
NOW TEST 
"""
#line by line version WM style:

test_string='CBOP\nC    !ROUTINE: SIZE.h\nC    !INTERFACE:\nC    include SIZE.h\nC    !DESCRIPTION: \\bv\nC     *==========================================================*\nC     | SIZE.h Declare size of underlying computational grid.\nC     *==========================================================*\nC     | The design here supports a three-dimensional model grid\nC     | with indices I,J and K. The three-dimensional domain\nC     | is comprised of nPx*nSx blocks (or tiles) of size sNx\nC     | along the first (left-most index) axis, nPy*nSy blocks\nC     | of size sNy along the second axis and one block of size\nC     | Nr along the vertical (third) axis.\nC     | Blocks/tiles have overlap regions of size OLx and OLy\nC     | along the dimensions that are subdivided.\nC     *==========================================================*\nC     \\ev\nC\nC     Voodoo numbers controlling data layout:\nC     sNx :: Number of X points in tile.\nC     sNy :: Number of Y points in tile.\nC     OLx :: Tile overlap extent in X.\nC     OLy :: Tile overlap extent in Y.\nC     nSx :: Number of tiles per process in X.\nC     nSy :: Number of tiles per process in Y.\nC     nPx :: Number of processes to use in X.\nC     nPy :: Number of processes to use in Y.\nC     Nx  :: Number of points in X for the full domain.\nC     Ny  :: Number of points in Y for the full domain.\nC     Nr  :: Number of points in vertical direction.\nCEOP\n      INTEGER sNx\n      INTEGER sNy\n      INTEGER OLx\n      INTEGER OLy\n      INTEGER nSx\n      INTEGER nSy\n      INTEGER nPx\n      INTEGER nPy\n      INTEGER Nx\n      INTEGER Ny\n      INTEGER Nr\n      PARAMETER (\n     &           sNx =  62,\n     &           sNy =  62,\n     &           OLx =   2,\n     &           OLy =   2,\n     &           nSx =   1,\n     &           nSy =   1,\n     &           nPx =   1,\n     &           nPy =   1,\n     &           Nx  = sNx*nSx*nPx,\n     &           Ny  = sNy*nSy*nPy,\n     &           Nr  =   1)\n\nC     MAX_OLX :: Set to the maximum overlap region size of any array\nC     MAX_OLY    that will be exchanged. Controls the sizing of exch\nC                routine buffers.\n      INTEGER MAX_OLX\n      INTEGER MAX_OLY\n      PARAMETER ( MAX_OLX = OLx,\n     &            MAX_OLY = OLy )\n\n'
print_out='\n'.join([f'{i}:{text}' for i,text in enumerate(test_string.split('\n'))])
first_query="""
In the string below, which lines have config parameter value assignments on them?
Format your answer in json format as a dictionary of parameter name: line number pairs
{config_string}
"""
first_query="""
In the string below, which lines have config parameter value assignments on them?
For each of those lines determine the smallest substring that must be edited to change the value.
Format your answer in json format as a dictionary of {{'parameter name': {{'line_number':line_number,'substring':substring}}}}
{config_string}"""
out = ask_gpt(first_query.format(config_string=print_out), model='gpt-4-1106-preview',response_format={"type": "json_object"})
config_options=json.loads(out)
second_queries="""
If I want to edit the value of {parameter} in this string which of the following should I edit?
Please submit you answer in json format as follows: {{'choice':choice}}
string: {string_line}
choices:{choices}
"""
second_queries="""
If I want to edit the value of {parameter} in this string which substring should I replace?
Please submit you answer in json format as follows: {{'substring':substring}}
string: {substring}
"""
edit_lines_prompts={}
out=json.loads(out)
test_string_lines=test_string.split('\n')
list_of_parameters=["sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "nPx", "nPy", "Nr"]
for i,key in enumerate(out):
    line=test_string_lines[out[key]]
    edit_lines_prompts[key]=second_queries.format(**{'parameter':key,'substring':line})
#'gpt-3.5-turbo-1106' 'gpt-4-1106-preview'
responses=process_ask_gpt_in_parallel(edit_lines_prompts.values(), edit_lines_prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
responses={key:json.loads(responses[key]) for key in responses.keys()}
config_options={}
for key in responses.keys():
    config_options[key]={'substring':responses[key]['substring'],'line_number':out[key]}
#or use some sort of comprehension..
#TODO: add options for different types of formats

#take integer, float, etc.. and convert to python types..
# def convert_type(config_options_dict):
#     for key in config_options_dict.keys():
#         option_type=config_options_dict[key]['type']
#         if 

#TODO: write code that takes file type information and function below to take user input and format it properly for file
#TODO: take convert_type, edit_lines and new formatting function. Take run command code and then create demo out of it. Maybe with streamlit app.
def edit_lines(config_string,config_options,user_parameters):
    """
    config_string is your config file in string format
    config options is a  dict that contain parameter_name keys and sub dicts with line_number, substring keys
    user_parameters is a dict of parameter value pairs
    """
    config_string_lines=config_string.split('\n')
    new_lines=[]
    for i,line in enumerate(config_string_lines):
        new_line=line
        for parameter in user_parameters:
            if parameter in list(config_options.keys()):
                if i == config_options[parameter]['line_number']:
                    pattern = r'\d+'
    
                    # Use re.search to find the numerical value in the string
                    match = re.search(pattern, new_line)
    
                    if match:
                        numerical_value = match.group()  # Convert the matched value to an integer
                        new_substring=config_options[parameter]['substring'].replace(numerical_value,str(user_parameters[parameter]))
                        new_line=new_line.replace(config_options[parameter]['substring'],new_substring)
                    else:
                        print("No numerical value found in the string.")
                        continue
                        
        new_lines.append(new_line)
            
    new_config_string='\n'.join(new_lines)
    return new_config_string
#TODO: test does generating summary help performance?

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


def extract_info_example(example_run_doc_1):
    part1_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_1.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(part1_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    part1_extract=json.loads(out)
    
    part2_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_2.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(part2_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    part2_extract=json.loads(out)
    
    part3_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_3.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(part3_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    part3_extract=json.loads(out)
    
    part4_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_4.format(library_name='MITGCM',article=example_run_doc_1)
    out = ask_gpt(part4_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    part4_extract=json.loads(out)
    
    return part1_extract,part2_extract,part3_extract,part4_extract

output_file_names= ['XC.001.001.data',
  'YC.001.001.data',
  'XG.001.001.data',
  'YG.001.001.data',
  'RC.data',
  'RF.data',
  'DXC.001.001.data',
  'DYC.001.001.data',
  'DXG.001.001.data',
  'DYG.001.001.data',
  'DRC.data',
  'DRF.data',
  'RAC.001.001.data',
  'RAS.001.001.data',
  'RAW.001.001.data',
  'RAZ.001.001.data',
  'hFacC.001.001.data',
  'hFacS.001.001.data',
  'hFacW.001.001.data',
  'Depth.001.001.data',
  'Rhoref.data',
  'PHrefC.data',
  'PHrefF.data',
  'PH.001.001.data',
  'PHL.001.001.data',
  'Eta.0000000000.001.001.data',
  'U.0000000000.001.001.data',
  'V.0000000000.001.001.data',
  'W.0000000000.001.001.data',
  'T.0000000000.001.001.data',
  'S.0000000000.001.001.data',
  'pickup.0000025920.001.001.data',
  'pickup.ckptA.001.001.data',
  'pickup.ckptB.001.001.data']




############ CHOOSE DOCS TO READ FOR FINDING CONFIG VARIABLES  ################
WHICH_DOC_TO_READ_CONFIG_PROMPT_TEMPLATE="""
Below are the summaries of articles in the documentation of the library {library_name}.
You need to extract configuration variables which can be edited in using this library from the documentation.
You will be able to read the full documentation next.
What are the next {n} articles you would like to read to find configuration variables?
Please place the articles in order of to be read first to to be read last.
Please format your answer in json format as follows: 
    {{"articles":[list of article numbers]}}

Here are the summaries: 
{summaries}

Begin!
"""
def which_docs_to_read_example(all_summaries):
    which_articles_prompt=WHICH_DOC_TO_READ_CONFIG_PROMPT_TEMPLATE.format(library_name='MITGCM',summaries=all_summaries,n=10)
    out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    articles_to_read=json.loads(out)['articles']
    return articles_to_read


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

#TODO: This part is expensive with gpt-4 - maybe prefilter with gpt-3.5, just asking if there are configurable parameters in the section..
#TODO: if it works write loop for all docs
def extract_from_doc_example(articles_to_read):
    rst_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.RST, chunk_size=4000, chunk_overlap=0
    )
    sub_docs=rst_splitter.split_text(docs_in_string_format[articles_to_read[0]])
    which_articles_prompts={j: EXTRACT_FROM_DOC_PROMPT_TEMPLATE.format(library_name='MITGCM',article=sub_doc) 
             for j,sub_doc in enumerate(sub_docs)}
    #'gpt-3.5-turbo-1106' 'gpt-4-1106-preview'
    responses=process_ask_gpt_in_parallel(which_articles_prompts.values(), which_articles_prompts.keys(), model='gpt-4-1106-preview',max_workers=8,response_format={"type": "json_object"}) 
    sorted_keys = sorted(responses.keys())
    responses = {key: responses[key] for key in sorted_keys} #TODO: catch end of generation errors if they happen..
    # which_articles_prompt=EXTRACT_FROM_DOC_PROMPT_TEMPLATE.format(library_name='MITGCM',article=docs_in_string_format[articles_to_read[0]])
    # out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    # parameters=json.loads(out)['parameters']
    #TODO: write function that filters output of model
    config_files=[]
    parameters={}
    for key in responses.keys():
        try:
            for out in json.loads(responses[key])['parameters']:
                if 'name' in out.keys():
                    if out['name'] not in parameters.keys():
                        parameters[out['name']]=out
                    else:
                        parameters[out['name']].update(out)
                if 'file' in out.keys():
                    config_files.append(out['file'])
                    
        except:
            print(responses[key])
            print(key)
    return responses
#TODO: get file names from outputs and then check against file names..
#TODO: combine parameter dicts from responses.

########## CHOOSE WHICH CODE DIRS TO LOOK AT #############

#TODO: need to find a way to identify the folders with configuration code in them..



################ CHOOSE WHICH CODE SUBDIRS TO LOOK INTO ################
WHICH_CODE_DIRS_PROMPT="""
Of the following folders (given with their name and the names of the files in them), which folders have files with configuration options in them?
{folders_print}
"""
#folders print looks like this - 
"""1: build - ['.gitignore']
2:code - ['CPP_EEOPTIONS.h_mpi','CPP_OPTIONS.h','OBCS_OPTIONS.h','packages.conf','SIZE.h','SIZE.h_mpi']
3:run - ['.gitignore']
4:results - ['output.txt']
5:input - ['data',
 'data.obcs',
 'data.pkg',
 'data.ptracers',
 'data.rbcs',
 'eedata',
 'eedata.mth',
 'gendata.m',
 'OBmeridU.bin',
 'OBzonalS.bin',
 'OBzonalU.bin',
 'OBzonalW.bin',
 'OB_EastH.bin',
 'OB_EastU.bin',
 'OB_WestH.bin',
 'OB_WestU.bin',
 'rbcs_mask.bin',
 'rbcs_Tr1_fld.bin',
 'topog.bump']
6: input-nlfs -  ['data', 'data.obcs', 'data.pkg', 'eedata', 'eedata.mth']"""

################### GET SUMMARIES OF CODE ################################
SUMMARIZE_CODE_FILE_PROMPT_TEMPLATE="""
Below is the content of an code file in the library {library_name}.
Can you please create a through and concise summary of the contents of the file including the purpose of the file, functions and classes found within the file, along with their purpose.
Please format your answer in json format as follows: 
    {{"summary":summary of code file}}

Here is the code: 
{code}

Begin!
"""

#get code file content
code_directories=['/media/hdd/Code/auto-config/MITgcm/verification/exp2/code',
                  '/media/hdd/Code/auto-config/MITgcm/verification/exp2/input',
                  '/media/hdd/Code/auto-config/MITgcm/verification/exp2/input.rigidLid']
unallowable_formats=['.bin']
code_in_string_format=[]
relevant_code_files=[]
for code_dir in code_directories:
    code_files=glob.glob(code_dir+'/**',recursive=True)
    for file_path in code_files:
        if not os.path.isdir(file_path) and os.path.splitext(file_path)[1] not in unallowable_formats:
            with open(file_path, 'r') as file:
                file_content = file.read()
            code_in_string_format.append(file_content)
            relevant_code_files.append(file_path)

def summarize_code_example(code_in_string_format):
    code_summary_prompts={j: SUMMARIZE_CODE_FILE_PROMPT_TEMPLATE.format(library_name='MITGCM',code=doc) 
              for j,doc in enumerate(code_in_string_format)}
    #'gpt-3.5-turbo-1106' 'gpt-4-1106-preview'
    code_responses=process_ask_gpt_in_parallel(code_summary_prompts.values(), code_summary_prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
    #TODO: handle issues of prompt being cut off with gpt-3.5
    sorted_keys = sorted(code_responses.keys())
    code_responses = {key: code_responses[key] for key in sorted_keys}
    code_summaries = ' '.join([f"Code {i} : {json.loads(code_responses[key])['summary']}\n\n" for i,key in enumerate(code_responses.keys())])
    return code_summaries

example_1="""
#ifndef CPP_OPTIONS_H
#define CPP_OPTIONS_H

CBOP
C !ROUTINE: CPP_OPTIONS.h
C !INTERFACE:
C #include "CPP_OPTIONS.h"

C !DESCRIPTION:
C *==================================================================*
C | main CPP options file for the model:
C | Control which optional features to compile in model/src code.
C *==================================================================*
CEOP

C CPP flags controlling particular source code features

C-- Forcing code options:

C o Shortwave heating as extra term in external_forcing.F
C Note: this should be a run-time option
#undef SHORTWAVE_HEATING

C o Include/exclude Geothermal Heat Flux at the bottom of the ocean
#undef ALLOW_GEOTHERMAL_FLUX

C o Allow to account for heating due to friction (and momentum dissipation)
#undef ALLOW_FRICTION_HEATING

C o Allow mass source or sink of Fluid in the interior
C   (3-D generalisation of oceanic real-fresh water flux)
#undef ALLOW_ADDFLUID

C o Include pressure loading code
#define ATMOSPHERIC_LOADING

C o Include/exclude balancing surface forcing fluxes code
#undef ALLOW_BALANCE_FLUXES

C o Include/exclude balancing surface forcing relaxation code
#undef ALLOW_BALANCE_RELAX

C o Include/exclude checking for negative salinity
#undef CHECK_SALINITY_FOR_NEGATIVE_VALUES

C-- Options to discard parts of the main code:

C o Exclude/allow external forcing-fields load
C   this allows to read & do simple linear time interpolation of oceanic
C   forcing fields, if no specific pkg (e.g., EXF) is used to compute them.
#undef EXCLUDE_FFIELDS_LOAD
C   If defined, use same method (with pkg/autodiff compiled or not) for checking
C   when to load new reccord ; by default, use simpler method with pkg/autodiff.
#undef STORE_LOADEDREC_TEST

C o Include/exclude phi_hyd calculation code
#define INCLUDE_PHIHYD_CALCULATION_CODE

C o Include/exclude sound speed calculation code
C o (Note that this is a diagnostic from Del Grasso algorithm, not derived
C    from EOS)
#undef INCLUDE_SOUNDSPEED_CALC_CODE

C-- Vertical mixing code options:

C o Include/exclude calling S/R CONVECTIVE_ADJUSTMENT
#define INCLUDE_CONVECT_CALL

C o Include/exclude calling S/R CONVECTIVE_ADJUSTMENT_INI, turned off by
C   default because it is an unpopular historical left-over
#undef INCLUDE_CONVECT_INI_CALL

C o Include/exclude call to S/R CALC_DIFFUSIVITY
#define INCLUDE_CALC_DIFFUSIVITY_CALL

C o Allow full 3D specification of vertical diffusivity
#undef ALLOW_3D_DIFFKR

C o Allow latitudinally varying BryanLewis79 vertical diffusivity
#undef ALLOW_BL79_LAT_VARY

C o Exclude/allow partial-cell effect (physical or enhanced) in vertical mixing
C   this allows to account for partial-cell in vertical viscosity and diffusion,
C   either from grid-spacing reduction effect or as artificially enhanced mixing
C   near surface & bottom for too thin grid-cell
#undef EXCLUDE_PCELL_MIX_CODE

C o Exclude/allow to use isotropic 3-D Smagorinsky viscosity as diffusivity
C   for tracers (after scaling by constant Prandtl number)
#undef ALLOW_SMAG_3D_DIFFUSIVITY

C-- Time-stepping code options:

C o Include/exclude combined Surf.Pressure and Drag Implicit solver code
#undef ALLOW_SOLVE4_PS_AND_DRAG

C o Include/exclude Implicit vertical advection code
#define INCLUDE_IMPLVERTADV_CODE

C o Include/exclude AdamsBashforth-3rd-Order code
#undef ALLOW_ADAMSBASHFORTH_3

C o Include/exclude Quasi-Hydrostatic Stagger Time-step AdamsBashforth code
#undef ALLOW_QHYD_STAGGER_TS

C-- Model formulation options:

C o Allow/exclude "Exact Convervation" of fluid in Free-Surface formulation
C   that ensures that d/dt(eta) is exactly equal to - Div.Transport
#define EXACT_CONSERV

C o Allow the use of Non-Linear Free-Surface formulation
C   this implies that grid-cell thickness (hFactors) varies with time
#undef NONLIN_FRSURF
C o Disable code for rStar coordinate and/or code for Sigma coordinate
c#define DISABLE_RSTAR_CODE
c#define DISABLE_SIGMA_CODE

C o Include/exclude nonHydrostatic code
#undef ALLOW_NONHYDROSTATIC

C o Include/exclude GM-like eddy stress in momentum code
#undef ALLOW_EDDYPSI

C-- Algorithm options:

C o Include/exclude code for Non Self-Adjoint (NSA) conjugate-gradient solver
#undef ALLOW_CG2D_NSA

C o Include/exclude code for single reduction Conjugate-Gradient solver
#define ALLOW_SRCG

C o Choices for implicit solver routines solve_*diagonal.F
C   The following has low memory footprint, but not suitable for AD
#undef SOLVE_DIAGONAL_LOWMEMORY
C   The following one suitable for AD but does not vectorize
#undef SOLVE_DIAGONAL_KINNER

C   Implementation alternative (might be faster on some platforms ?)
#undef USE_MASK_AND_NO_IF

C-- Retired code options:

C o ALLOW isotropic scaling of harmonic and bi-harmonic terms when
C   using an locally isotropic spherical grid with (dlambda) x (dphi*cos(phi))
C *only for use on a lat-lon grid*
C   Setting this flag here affects both momentum and tracer equation unless
C   it is set/unset again in other header fields (e.g., GAD_OPTIONS.h).
C   The definition of the flag is commented to avoid interference with
C   such other header files.
C   The preferred method is specifying a value for viscAhGrid or viscA4Grid
C   in data which is then automatically scaled by the grid size;
C   the old method of specifying viscAh/viscA4 and this flag is provided
C   for completeness only (and for use with the adjoint).
c#define ISOTROPIC_COS_SCALING

C o This flag selects the form of COSINE(lat) scaling of bi-harmonic term.
C *only for use on a lat-lon grid*
C   Has no effect if ISOTROPIC_COS_SCALING is undefined.
C   Has no effect on vector invariant momentum equations.
C   Setting this flag here affects both momentum and tracer equation unless
C   it is set/unset again in other header fields (e.g., GAD_OPTIONS.h).
C   The definition of the flag is commented to avoid interference with
C   such other header files.
c#define COSINEMETH_III

C o Use "OLD" UV discretisation near boundaries (*not* recommended)
C   Note - only works with pkg/mom_fluxform and "no_slip_sides=.FALSE."
C          because the old code did not have no-slip BCs
#undef OLD_ADV_BCS

C o Use LONG.bin, LATG.bin, etc., initialization for ini_curviliear_grid.F
C   Default is to use "new" grid files (OLD_GRID_IO undef) but OLD_GRID_IO
C   is still useful with, e.g., single-domain curvilinear configurations.
#undef OLD_GRID_IO

C o Use old EXTERNAL_FORCING_U,V,T,S subroutines (for backward compatibility)
#undef USE_OLD_EXTERNAL_FORCING

C-- Other option files:

C o Execution environment support options
#include "CPP_EEOPTIONS.h"

C o Include/exclude single header file containing multiple packages options
C   (AUTODIFF, COST, CTRL, ECCO, EXF ...) instead of the standard way where
C   each of the above pkg get its own options from its specific option file.
C   Although this method, inherited from ECCO setup, has been traditionally
C   used for all adjoint built, work is in progress to allow to use the
C   standard method also for adjoint built.
c#ifdef PACKAGES_CONFIG_H
c# include "ECCO_CPPOPTIONS.h"
c#endif

#endif /* CPP_OPTIONS_H */
"""

def summarize_code_example_2(example_1):
    summarize_code_prompt=SUMMARIZE_CODE_FILE_PROMPT_TEMPLATE.format(code=example_1)
    out = ask_gpt(summarize_code_prompt, model='gpt-3.5-turbo-1106',response_format={"type": "json_object"})
    return out

###################### CHOOSE WHICH CODE FILES TO EXAMINE #######################
WHICH_CODE_TO_READ_PROMPT_TEMPLATE="""
Below are the summaries of code files in the library {library_name}.
You need to extract configuration variables which can be edited in using this library from the code files.
You will be able to read the full code files next.
What are the next {n_code_files} code files you would like to read to find configuration variables?
Please place the code files in order of to be read first to to be read last.
Please format your answer in json format as follows: 
    {{"code_files":[list of code file numbers]}}

Here are the summaries: 
{summaries}

Begin!
"""

def choose_code_example(code_summaries):
    which_code_prompt=WHICH_CODE_TO_READ_PROMPT_TEMPLATE.format(library_name='MITGCM',summaries=code_summaries,n_code_files=10)
    out = ask_gpt(which_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    code_to_read=json.loads(out)['code_files']
    return code_to_read


################ EXTRACT VARIABLES FROM CODE ################

EXTRACT_OPTIONS_FROM_CODE_FILE_PROMPT_TEMPLATE="""
Below is the content of an code file in the library {library_name}.
Can you please extract information about any options within the file. 

Please format your answer in json format as follows: 
    {{"options":[list of dictionaries about options]}}
    
Each dictionary about a configuration variable can have the following keys:
    {{"name": name of the option,
      "line_content":full content of the line the option can be found on,
      "description":description of the option}}

Here is the code: 
{code}

Begin!
"""

def extract_code_example():
    extract_code_prompt=EXTRACT_OPTIONS_FROM_CODE_FILE_PROMPT_TEMPLATE.format(library_name='MITGCM',code=example_1)
    out = ask_gpt(extract_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    options=json.loads(out)
    return options
    
#TODO: better method for variable changing automatically - 
#basically for now the idea is to find out information about the code, language, file type, etc.. 
#and use that and the content of the line to find the line and then find the appropriate method of switching the option
#then based on the options manually added below we will force gpt to use one of a few types or if using a gui, add widgets which enforce this..

def change_option(name,line_content,new_option,code_file_content):
    def find_line_number(code_file_content,line_content):
        """
        example usage:
        line_number=find_line_number(example_1,"#undef USE_OLD_EXTERNAL_FORCING")

        """
        lines=code_file_content.split('\n')
        for i,line in enumerate(lines):
            if line_content in line:
                #TODO: make sure there aren't too lines with the content in there
                return i,lines
        return 'Line Not Found',lines
    
    line_number,code_lines=find_line_number(code_file_content,line_content)
    
    if '#' in line_content:
        if new_option=='true':
            new_line_content=line_content.replace('#','')
        else:
            new_line_content=line_content
            
    code_lines[line_number]=new_line_content
    return '\n'.join(code_lines)
    
    #TODO: make sure that all the is_simulation_file results are the same..
#TODO: right now integrated pipeline uses subset of operations for demo purposes, to use for real, expand from single example to parallel calls over whole list/step 3 tasks (given gpt-4 rate limits)


def integrated_pipeline(docs_directory='/media/hdd/Code/auto-config/MITgcm/doc',code_directory='/media/hdd/Code/auto-config/MITgcm'):
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
    prompts={j: SUMMARIZE_DOCUMENTATION_DOC_PROMPT_TEMPLATE.format(library_name='MITGCM',article=doc) 
             for j,doc in enumerate(docs_in_string_format)}
    summarize_docs_time=time.time()
    responses=process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
    print('Took this much time to summarize the docs - ',time.time()-summarize_docs_time)
    sorted_keys = sorted(responses.keys())
    responses = {key: responses[key] for key in sorted_keys}
    all_summaries = ' '.join([f"Article {i} : {json.loads(responses[key])['summary']}\n\n" for i,key in enumerate(responses.keys())])
    
    #############   STEP 2: WHICH DOCS TO READ  ####################
    
    which_articles_prompt=WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE.format(library_name='MITGCM',summaries=all_summaries,n=20)
    choose_docs_time=time.time()
    out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to choose which docs to read - ',time.time()-choose_docs_time)
    run_articles_to_read=json.loads(out)['articles']
    
    #############   STEP 3: GET CONFIG PARAMS FROM DOCS/CODE  ####################
    
       ########   STEP 3a: GET CONFIG FILES FROM DOCS  #############
       
    example_run_doc_1=docs_in_string_format[12]
    part12_prompt=GET_CONFIG_FILES_FROM_DOCS_PROMPT_TEMPLATE.format(library_name='MITGCM',article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(part12_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config files from docs - ',time.time()-read_doc_time)
    part12_extract=json.loads(out)   
    modification_code_files=part12_extract['modification_code_files']
    
       ########   STEP 3b: GET CONFIG PARAMS FROM DOC/CODE PAIRS  #############
    # 2 examples
    
    doc_code_prompt=GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT.format(library_name='MITGCM',code_file_name=extracted_file_names[0],article=example_run_doc_1,code=example_run_code_1_file_1) #example_run_code_1_file_1
    read_doc_time=time.time()
    out = ask_gpt(doc_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config params from doc/code pair - ',time.time()-read_doc_time)
    docs_code_extract_1=json.loads(out)
    
    doc_code_prompt=GET_CONFIG_PARAMS_FROM_DOCS_AND_CODE_PROMPT.format(library_name='MITGCM',code_file_name=extracted_file_names[1],article=example_run_doc_1,code=example_run_code_1_file_2) #example_run_code_1_file_2
    read_doc_time=time.time()
    out = ask_gpt(doc_code_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get config params from doc/code pair - ',time.time()-read_doc_time)
    docs_code_extract=json.loads(out)
    
    #############   STEP 4: GET OUTPUT VARIABLES FROM OUTPUT FILES/DOCS  ####################

    output_file_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_4.format(library_name='MITGCM',article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(output_file_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get output files from docs - ',time.time()-read_doc_time)
    output_file_extract=json.loads(out)
    output_files=output_file_extract['output_file_names']
    output_variables=output_file_extract['output_variable_names']
    
    
    #############   STEP 5: GET RUN COMMANDS  ####################
    
    run_commands_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_3.format(library_name='MITGCM',article=example_run_doc_1)
    read_doc_time=time.time()
    out = ask_gpt(run_commands_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
    print('Took this much time to get run commands from docs - ',time.time()-read_doc_time)
    run_commands_extract=json.loads(out)
    run_commands=run_commands_extract['run_instructions']
                

