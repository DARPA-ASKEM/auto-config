# -*- coding: utf-8 -*-
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.schema import Document
import os
from llm_tools import process_ask_gpt_in_parallel,parse_json_response,ask_gpt,ask_gpt_with_continue
import glob
import json
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
import os
import time
from CodeLATS.code_lats import use_lats
from main import *
#TODO: could try giving forums as docs to look up stuff later - https://forum.mmm.ucar.edu/
#TODO: add tutorials? 
#TODO: get compilation or installation instructions
"""https://www2.mmm.ucar.edu/wrf/users/tutorial/tutorial_presentations_2021.htm 
https://www2.mmm.ucar.edu/wrf/users/tutorial/tutorial.html
https://www2.mmm.ucar.edu/wrf/users/tutorial/tutorial.html
https://www2.mmm.ucar.edu/wrf/users/docs/AFWA_Diagnostics_in_WRF.pdf
"""
def get_wrf_docs():
    #user's guide- https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/index.html
    #TODO: write way to get all the html files automatically
    html_files=["https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/index.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/overview.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/compiling.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/initialization.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/running_wrf.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/dynamics.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/physics.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/namelist_variables.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/output.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/software.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/post_processing.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/utilities_tools.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wrfda.html",
                "https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/fire.html"]
    
    loader = AsyncHtmlLoader(html_files)
    docs = loader.load()
    #split by section.. <section id=  rm  - " "> #rm - </section>
    def find_first_two_quotes(s):
        indices = []
        for i, char in enumerate(s):
            if char == '"':
                indices.append(i)
                if len(indices) == 2:
                    break
        return indices
    
    section_docs=[]
    for doc in docs:
        sections=doc.page_content.split('<section id=')
        section_docs.append(Document(page_content=sections[0],metadata={'source':doc.metadata['source'],
                                                                        'title':doc.metadata['title'] + ' - Intro',
                                                                        "language":doc.metadata['language']}))
        if len(sections)>1:
            for i in range(1, len(sections)):
                name_indices=find_first_two_quotes(sections[i])
                name = sections[i][name_indices[0]+1:name_indices[1]]
                #remove id and end section tag
                sections[i]=sections[i][name_indices[1]+2:]
                sections[i]=sections[i].replace('</section>','')
                section_docs.append(Document(page_content=sections[i],metadata={'source':doc.metadata['source'],
                                                                                'title':doc.metadata['title'] + f' - {name}',
                                                                                "language":doc.metadata['language']}))
                
    #then process html2text
    
    from langchain_community.document_transformers import Html2TextTransformer
    
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(section_docs)
    folder_name='/media/hdd/Code/auto-config/WRF/external_docs'
    for doc in docs_transformed:
        file_name=os.path.join(folder_name,f"{doc.metadata['title']}.txt")
        print(file_name)
        with open(file_name,'w') as f:
            f.write(doc.page_content)
        f.close()



def wrf_demo_pipeline(docs_directory='/media/hdd/Code/auto-config/WRF/external_docs',code_directory='/media/hdd/Code/auto-config/WRF/test',library_name='WRF'):#get model config stuff related to em_b_wave
        doc_files=glob.glob(docs_directory+'/**',recursive=True)
        docs_in_string_format=[]
        valid_doc_files=[]
        for file_path in doc_files:
            if '.txt' in file_path:
                with open(file_path, 'r') as file:
                    file_content = file.read()
                docs_in_string_format.append(file_content)
                valid_doc_files.append(file_path)
        print(len(valid_doc_files))
        
        #############   STEP 1: get summaries of documentation articles ####################
        prompts={j: SUMMARIZE_DOCUMENTATION_DOC_PROMPT_TEMPLATE.format(library_name=library_name,article=doc) 
                 for j,doc in enumerate(docs_in_string_format)}
        summarize_docs_time=time.time()
        responses=process_ask_gpt_in_parallel(prompts.values(), prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=24,response_format={"type": "json_object"}) 
        print('Took this much time to summarize the docs - ',time.time()-summarize_docs_time)
        sorted_keys = sorted(responses.keys())
        responses = {key: responses[key] for key in sorted_keys}
        all_summaries = ' '.join([f"Article {i} :\n Title: {valid_doc_files[i]} \n Summary: {json.loads(responses[key])['summary']}\n\n" for i,key in enumerate(responses.keys())])
        print(responses[0],'\n',responses[1],'\n',responses[2])
        
        #############   STEP 2: WHICH DOCS TO READ  ####################
        which_articles_prompt=WHICH_DOC_TO_READ_RUN_PROMPT_TEMPLATE.format(library_name=library_name,summaries=all_summaries,n=20,specific_request='Specifically you are trying to run idealized simulations')#'Specifically you are trying to run idealized simulations')
        choose_docs_time=time.time()
        out = ask_gpt(which_articles_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
        print('Took this much time to choose which docs to read - ',time.time()-choose_docs_time)
        run_articles_to_read=json.loads(out)['articles']
        print(run_articles_to_read)
        
        #############   STEP 3: GET CONFIG PARAMS FROM DOCS/CODE  ####################
           ########   STEP 3a: GET CONFIG FILES FROM DOCS  #############
        
        example_run_doc_1=docs_in_string_format[37] 
        part12_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_12.format(library_name=library_name,article=example_run_doc_1)
        read_doc_time=time.time()
        out = ask_gpt(part12_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
        print('Took this much time to get config files from docs - ',time.time()-read_doc_time)
        part12_extract=json.loads(out)   
        modification_code_files=part12_extract['modification_code_files']
        print(modification_code_files)
        
            ########   STEP 3b: GET CONFIG PARAMS FROM CODE  #############
        extracted_file_names=['WRF/test/em_b_wave/namelist.input', 'WRF/run/README.namelist']
        with open(os.path.join(code_directory,'em_b_wave/namelist.input'),'r') as f:
            example_run_code_1_file_1=f.read()
        read_code_time=time.time()
        conf_result_1=ask_gpt(get_config_simple_1.format(config_text=example_run_code_1_file_1),'gpt-3.5-turbo-1106',response_format={"type": "json_object"})
        print('Took this much time to get config params from code - ',time.time()-read_code_time)
        variables=json.loads(conf_result_1)['configuration_variables']
        print(variables)
        
            ########   STEP 3c: GET CONFIG DETAILS FROM CODE   #############
        get_details_prompts={i:get_config_simple_2_with_config.format(variables='\n'.join(variables[i:i+10]),documentation=example_run_code_1_file_1) for i in range(0,len(variables),10)}
        detailed_responses=process_ask_gpt_in_parallel(get_details_prompts.values(), get_details_prompts.keys(), model='gpt-3.5-turbo-1106',max_workers=8,response_format={"type": "json_object"}) 
        detailed_responses={key:json.loads(detailed_responses[key]) for key in detailed_responses.keys()}
        variable_details={}
        for key in detailed_responses.keys():
            for d in detailed_responses[key]['configuration_variable_details']:
                name=d['variable_name']
                d.pop('variable_name')
                variable_details[name]=d
        variable_details
                
        ########   STEP 4: CREATE CONFIG MODIFICATION FUNCTIONS FROM CODE  #############    
        function_time=time.time()
        functions=get_modification_functions(variables,example_run_code_1_file_1)
        print('Took this much time to get functions to modify config - ',time.time()-function_time)
        
        #############   STEP 5: GET OUTPUT VARIABLES FROM OUTPUT FILES/DOCS  ####################

        output_file_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_4.format(library_name=library_name,article=example_run_doc_1) 
        read_doc_time=time.time()
        out = ask_gpt(output_file_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
        print('Took this much time to get output files from docs - ',time.time()-read_doc_time)
        output_file_extract=json.loads(out)
        output_files=output_file_extract['output_file_names']
        output_variables=output_file_extract['output_variable_names']
        
        
        
        #############   STEP 6: GET RUN COMMANDS  ####################
        
        run_commands_prompt=HOW_TO_RUN_PROMPT_TEMPLATE_part_3.format(library_name=library_name,article=example_run_doc_1) 
        read_doc_time=time.time()
        out = ask_gpt(run_commands_prompt, model='gpt-4-1106-preview',response_format={"type": "json_object"})
        print('Took this much time to get run commands from docs - ',time.time()-read_doc_time)
        run_commands_extract=json.loads(out)
        run_commands=run_commands_extract['run_instructions']
        run_commands
        
        ########### MODIFY CODE #################
        new_config=use_generate_functions_to_modify_config(functions,variable_details,{'run_days':7,'dx':200000,'dy':200000},example_run_code_1_file_1)
        with open(os.path.join(code_directory,'em_b_wave/namelist.input'),'w') as f:
            f.write(new_config)
        f.close()
        ############### RUN CODE ###################
        print(run_commands)
        