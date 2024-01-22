
## Introduction
This repository contains a robust and flexible conversational agent system designed to process and respond to various commands and tasks. The system is built around three main types of agents: BaseAgent, ReactAgent, and MasterAgent.

**BaseAgent**: This is the foundational agent class that handles the basic operations such as fetching models, setting prompts, and managing agent messages and responses.

**ReactAgent**: This agent extends the BaseAgent and includes additional functionality for processing commands. It uses a list of tools (functions) to perform various actions based on the commands it receives.

**RAG Agent**: This agent extends BaseAgent and generate response based on results of semantic search from vectorstore

**MasterAgent**: This agent extends the ReactAgent and is designed to manage other agents. It can fetch other agents by their names and start tasks on them. It also includes a gatekeeper function that can validate user input.

The system also includes a Context class that holds the application context, which is used by the agents to maintain state and share information.

## Usage
To use these agents, you need to initialize them with the appropriate parameters. Here's a basic example of how to use these agents:

from agent_system import MasterAgent, ReactAgent, BaseAgent

### Initialize a BaseAgent
```python
from analysta_llm_agents import BaseAgent

base_agent = BaseAgent(agent_prompt="your_prompt")
```

### Initialize a ReactAgent
```python
from analysta_llm_agents import ReactAgent

react_agent = ReactAgent(agent_prompt="your_prompt", actions=[...])
```

#### Actions

The actions for the ReactAgent are essentially a list of functions that the agent can perform based on the commands it receives. Each action is a Python function that takes certain parameters and returns a result.

Here's a basic structure of an action function:
```python
from analysta_llm_agents.tools.tool import tool

@tool
def action_name(ctx, arg1: str, arg2: str, ...):
    # Perform some operation using the arguments and context
    ...
    # Return the result
    return result
```

The context `ctx` parameter is an instance of the Context class and is used to share information between different parts of the application. The other parameters (arg1, arg2, etc.) are the arguments required by the action.

These action functions are added to the ReactAgent during initialization:

```python

actions = [action1, action2, ...]
react_agent = ReactAgent(agent_prompt="your_prompt", actions=actions)
```
When the ReactAgent receives a command, it looks for a corresponding action in its list of actions and executes it. The arguments for the action are taken from the command. If the action requires a context, the current context of the agent is passed as an argument.

Please note that the action functions should be designed to handle errors and exceptions appropriately to ensure the robustness of the agent.

### Initialize a MasterAgent
```python
from analysta_llm_agents import MasterAgent

master_agent = MasterAgent(agent_prompt="your_prompt", agents=[...])
```

#### Agents
TO DO


## Start a task on the master agent
```python
for message in master_agent.start("your_task"):
    print(message)
```

Please note that the start method is a generator function and you need to iterate over it to get the responses.

## LLM Configuration
Still in to do


## Build
1. `python3 -m pip install --upgrade build`
2. `python3 -m build `