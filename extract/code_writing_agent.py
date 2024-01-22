# -*- coding: utf-8 -*-
import argparse
import shutil
import os
import warnings
import logging

#######langchain #########
from langchain.prompts import PromptTemplate
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.cache import SQLiteCache
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ChatMessageHistory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor,StructuredChatAgent
#######langchain #########
from keys import langsmith_jataware
from langchain_experimental.tools.python.tool import PythonREPLTool

test_string='CBOP\nC    !ROUTINE: SIZE.h\nC    !INTERFACE:\nC    include SIZE.h\nC    !DESCRIPTION: \\bv\nC     *==========================================================*\nC     | SIZE.h Declare size of underlying computational grid.\nC     *==========================================================*\nC     | The design here supports a three-dimensional model grid\nC     | with indices I,J and K. The three-dimensional domain\nC     | is comprised of nPx*nSx blocks (or tiles) of size sNx\nC     | along the first (left-most index) axis, nPy*nSy blocks\nC     | of size sNy along the second axis and one block of size\nC     | Nr along the vertical (third) axis.\nC     | Blocks/tiles have overlap regions of size OLx and OLy\nC     | along the dimensions that are subdivided.\nC     *==========================================================*\nC     \\ev\nC\nC     Voodoo numbers controlling data layout:\nC     sNx :: Number of X points in tile.\nC     sNy :: Number of Y points in tile.\nC     OLx :: Tile overlap extent in X.\nC     OLy :: Tile overlap extent in Y.\nC     nSx :: Number of tiles per process in X.\nC     nSy :: Number of tiles per process in Y.\nC     nPx :: Number of processes to use in X.\nC     nPy :: Number of processes to use in Y.\nC     Nx  :: Number of points in X for the full domain.\nC     Ny  :: Number of points in Y for the full domain.\nC     Nr  :: Number of points in vertical direction.\nCEOP\n      INTEGER sNx\n      INTEGER sNy\n      INTEGER OLx\n      INTEGER OLy\n      INTEGER nSx\n      INTEGER nSy\n      INTEGER nPx\n      INTEGER nPy\n      INTEGER Nx\n      INTEGER Ny\n      INTEGER Nr\n      PARAMETER (\n     &           sNx =  62,\n     &           sNy =  62,\n     &           OLx =   2,\n     &           OLy =   2,\n     &           nSx =   1,\n     &           nSy =   1,\n     &           nPx =   1,\n     &           nPy =   1,\n     &           Nx  = sNx*nSx*nPx,\n     &           Ny  = sNy*nSy*nPy,\n     &           Nr  =   1)\n\nC     MAX_OLX :: Set to the maximum overlap region size of any array\nC     MAX_OLY    that will be exchanged. Controls the sizing of exch\nC                routine buffers.\n      INTEGER MAX_OLX\n      INTEGER MAX_OLY\n      PARAMETER ( MAX_OLX = OLx,\n     &            MAX_OLY = OLy )\n\n'
#TODO: try - https://github.com/robotpy/cxxheaderparser 
def test_python_repl_tool(test_string):
    tool=PythonREPLTool()
    tool.description='''A Python shell. The python environment has an global variable called test string already instantiated.
    Use this to test that your function works properly by running your the test on the test string variable. The test string is the content of the code file you were given.
    Make sure to start all scripts with global test_string;
    If you want to see the output of a value, you should print it out with `print(...)`.'''
    tool.python_repl.globals['test_string'] = test_string
    return tool

def test_function_tool(test_string):
    #give raw page source and let it manipulate using python and selenium
    #TODO:write function
    tool=PythonREPLTool()
    tool.description='''A tool to test your function using a test you generate on the given code string.
    Use this to test that your function works properly before submitting your final answer.
    The input to this tool should be the full function definition of the edit_parameters function and a test for the edit_parameters function. 
    The test string is the content of the code file you were given.
    The tool will print what is returned by your test so format your test accordingly.'''
    tool.python_repl.globals['test_string'] = test_string
    return tool
#TODO: give related functions to the code agent when it fails...

#TODO: maybe just make this a tool that executes the test using the given function and returns the output..
#TODO:provide examples..
tools = []
tools.append(test_python_repl_tool(test_string))
langchain.verbose = True
langchain.debug = True
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = langsmith_jataware
os.environ['LANGCHAIN_PROJECT'] = 'autoconfig-debug'
os.environ['LANGCHAIN_CALLBACKS_BACKGROUND'] = 'true'
llm = ChatOpenAI(model_name='gpt-4-1106-preview',temperature=0)
#add final answer constraint
#TODO: maybe specify I/O of the function and test function and what the test values should be?
#DONE: prompt the model by asking a question about how to do the coding required? (ie explicitly make it reason before code gen?) - did not really work
#TODO: could I just ask it to write a value assignment line and just place it at the top or replace the old line?
prompt={}
prompt['prefix']="""
You will be given the contents of a code file which contains a parameter to be edited for configuring a simulation from the library MITGCM which is written in fortran.
The name of the parameter,the code file name and the code file content will be provided below.
Please write a short python function named edit_parameters which can take the string content of the file below and change the given parameter to a valid value.
Please include a detailed doc string and include types for the inputs. These inputs should include the configuration string and dictionary of parameter,new value pairs for each parameter to be updated.
Please also write a test which accepts a text string as an input variable and that confirms that the function works properly on the given code string below. Run that test in the repl before submitting your answer.

Please use the python_repl code to check that your code works properly before submitting your final answer.
Your final should ONLY include the final function.

The tools you have access to are:
"""
prompt['prefix']= """
You will be given the contents of a code file which contains a parameter to be edited for configuring a simulation from the library MITGCM which is written in fortran.
But first I am going to ask you some questions.
To answer these questions you should not need to use your tools, you will need them later.

"""

prompt['format_instructions']="""\
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Instruction: Instructions from the user
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""
prompt['suffix']="""Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.

{chat_history}
{input}

{agent_scratchpad}"""

prompt = StructuredChatAgent.create_prompt(
tools,
prefix=prompt['prefix'],
suffix=prompt['suffix'],
format_instructions=prompt['format_instructions'],
input_variables=["input", "chat_history", "agent_scratchpad","workflow"],
)

memory = ConversationBufferWindowMemory(memory_key="chat_history",output_key="output",k=5,chat_memory=ChatMessageHistory())

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True,
    memory=memory,return_intermediate_steps=True,
    handle_parsing_errors=True, max_iterations=25
)

inp=f"""
The name of the parameter to be edited by the function is sNx.
The name of the codefile is SIZE.h
The content of the code file is below:
    {test_string}
"""
# inp=f"""
# The names of the parameters to be edited by the function are sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nr
# The name of the codefile is SIZE.h
# The content of the code file is below:
#     {test_string}
# """

questions="""
First question. How would you write a python function to change the content of a configuration file? What would you possible need to change in a configuration file?
"""
questions2="""
Next question. What if the configuration file in questions was a .h file for configuring a fortran simulation?
"""
questions3=f"""
Now it is time to use your tools. 
Please write a short python function named edit_parameters which can take the string content of the file below and change the given parameter to a valid value.
Please include a detailed doc string and include types for the inputs.
Please also write a test which accepts a text string as an input variable and that confirms that the function works properly on the given code string below. Run that test in the repl before submitting your answer.
The name of the parameter,the code file name and the code file content will be provided below.

Please use the python_repl code to check that your code works properly before submitting your final answer.
Your final should ONLY include the final function.
{inp}
"""

examples_1="""
Here are some examples of similar python functions which parse config strings:
Config String: "max_connections=100; min_connections=5; timeout=30"
Python Function:
def parse_and_update_config_1(config_str, param, new_value):
    config_items = config_str.split('; ')
    updated_items = [f"{param}={new_value}" if item.startswith(param) else item for item in config_items]
    return '; '.join(updated_items)    

Config String: "servers=[server1, server2, server3]; mode=active"
Python Function:
def parse_and_update_config_2(config_str, param, new_value):
    config_items = config_str.split('; ')
    updated_items = [f"{param}={new_value}" if item.startswith(param) else item for item in config_items]
    return '; '.join(updated_items)

Config String: "path=/usr/local/bin; version=1.2.3; debug=True"    
Python Function:
def parse_and_update_config_3(config_str, param, new_value):
    config_items = config_str.split('; ')
    updated_items = [f"{param}={new_value}" if item.startswith(param) else item for item in config_items]
    return '; '.join(updated_items)

Config String: "color=blue; size=medium; layout=grid"
Python Function:
def parse_and_update_config_4(config_str, param, new_value):
    config_items = config_str.split('; ')
    updated_items = [f"{param}={new_value}" if item.startswith(param) else item for item in config_items]
    return '; '.join(updated_items)

Config String: "user=admin; password=12345; encryption=sha256"
Python Function:
def parse_and_update_config_5(config_str, param, new_value):
    config_items = config_str.split('; ')
    updated_items = [f"{param}={new_value}" if item.startswith(param) else item for item in config_items]
    return '; '.join(updated_items)
"""

examples_2="""
Config String:
[Database]
host=localhost
user=root
password=secret

Python Function:
import re
def parse_and_update_config_1(config_str, section, param, new_value):
    pattern = f"(\[{section}\][\s\S]*?){param}=.*"
    replacement = f"\\1{param}={new_value}"
    return re.sub(pattern, replacement, config_str, flags=re.MULTILINE)

Config String:
[Server]
ip=192.168.1.1
port=8080
[Client]
ip=192.168.1.2
port=9090

Python Function:

import re
def parse_and_update_config_2(config_str, section, param, new_value):
    pattern = f"(\[{section}\][\s\S]*?){param}=.*"
    replacement = f"\\1{param}={new_value}"
    return re.sub(pattern, replacement, config_str, flags=re.MULTILINE)

Config String:
[Paths]
root=/var/www
logs=/var/logs
[Environment]
debug=True
mode=production

Python Function:
import re
def parse_and_update_config_3(config_str, section, param, new_value):
    pattern = f"(\[{section}\][\s\S]*?){param}=.*"
    replacement = f"\\1{param}={new_value}"
    return re.sub(pattern, replacement, config_str, flags=re.MULTILINE)

Config String:
[User]
name=admin
access_level=high
[Security]
encryption=enabled
timeout=30

Python Function:
import re

def parse_and_update_config_4(config_str, section, param, new_value):
    pattern = f"(\[{section}\][\s\S]*?){param}=.*"
    replacement = f"\\1{param}={new_value}"
    return re.sub(pattern, replacement, config_str, flags=re.MULTILINE)

Config String:
[Graphics]
resolution=1920x1080
fullscreen=True
[Audio]
volume=80
mute=False

Python Function:
import re
def parse_and_update_config_5(config_str, section, param, new_value):
    pattern = f"(\[{section}\][\s\S]*?){param}=.*"
    replacement = f"\\1{param}={new_value}"
    return re.sub(pattern, replacement, config_str, flags=re.MULTILINE)
"""

examples_3="""
Example Configuration String:
# MaxIterations: 500
# Description: Sets the maximum number of iterations for the simulation
Python Function to Parse and Modify:

import re

def modify_max_iterations(config_string, new_value):
    pattern = r"(# MaxIterations: )(\d+)"
    replacement = f"\\1{new_value}"
    return re.sub(pattern, replacement, config_string)
Example Configuration String:
# TempThreshold: 350.0
# Description: Temperature threshold in Kelvin
Python Function to Parse and Modify:
import re

def modify_temp_threshold(config_string, new_value):
    pattern = r"(# TempThreshold: )(\d+\.\d+)"
    replacement = f"\\1{new_value}"
    return re.sub(pattern, replacement, config_string)
Example Configuration String:
# SimulationMode: Dynamic
# Description: Type of simulation mode (Static/Dynamic)
Python Function to Parse and Modify:
import re

def modify_simulation_mode(config_string, new_value):
    pattern = r"(# SimulationMode: )(\w+)"
    replacement = f"\\1{new_value}"
    return re.sub(pattern, replacement, config_string)
Example Configuration String:
# OutputFrequency: 100
# Description: Frequency of simulation output
Python Function to Parse and Modify:
import re

def modify_output_frequency(config_string, new_value):
    pattern = r"(# OutputFrequency: )(\d+)"
    replacement = f"\\1{new_value}"
    return re.sub(pattern, replacement, config_string)
Example Configuration String:
# EnableLogging: False
# Description: Enable or disable logging
Python Function to Parse and Modify:
import re

def modify_enable_logging(config_string, new_value):
    pattern = r"(# EnableLogging: )(\w+)"
    replacement = f"\\1{new_value}"
    return re.sub(pattern, replacement, config_string)
"""
agent_chain(questions)