## Analysta LLM Agents 🤖

Welcome to the Agent Framework repository! This robust Python library is engineered for developers looking to craft cutting-edge agents powered by large language models (LLMs). Dive into a world where creating intelligent, customizable agents is streamlined and efficient.

### Key Features 🛠

- **Custom Agent Profiles:** Tailor your agents with distinct roles, skills, and objectives to fit diverse applications. 🎨
- **Advanced Memory Systems:** Implements sophisticated short-term and long-term - memory functionalities to enhance agent performance and learning. 🧠
- **Modular Tool Integration:** Seamlessly incorporate a variety of tools, contractors, and data sources to expand your agent's capabilities. 🔧
- **Flexible Communication:** Craft and manage dynamic interactions between your agent and users, with support for complex command and response structures. 💬
- **Efficient Error Handling:** Robust error management ensures your agent remains resilient and responsive under various conditions. 🛡️

## How-Tos

### How to Create Custom Agent Profiles 🛠️

Creating custom agent profiles allows you to define unique characteristics and capabilities tailored to your specific needs. Here's a step-by-step guide to get you started:

#### Step 1: Define Your Agent's Role and Skills
Start by outlining the role, skills, and purpose of your agent. Consider what tasks your agent should perform and the knowledge it requires.

```python
agent_role = "Customer Support Assistant"
agent_skills = ["Handle FAQs", "Ticket Routing", "User Feedback Collection"]
agent_purpose = "To assist users with common queries and direct complex issues to the appropriate teams."
agent_constraints = """
- Adhere strictly to user privacy and data protection standards.
- Maintain a polite and professional tone in all interactions.
- Limit response length to ensure concise and relevant answers.
- Avoid speculative answers in areas outside of defined expertise.
- Escalate issues beyond the agent's capability to a human operator.
"""
```
#### Step 2: Initialize Your Agent
Use the Agent class to create an instance of your agent. Pass the role, skills, and any other specific parameters you defined.

```python
from analysta_llm_agents.agents.agent import Agent

class CustomAgent(Agent):
    
    def __init__(self, ctx, **kwargs):
        super.__init__(
            agent_prompt=f"{agent_role} skilled in {', '.join(agent_skills)}. {agent_purpose}",
            agent_constraints=agent_constraints
            llm_model_type="AzureChatOpenAI",  # This is basucally name of langchain classes
            llm_model_params={...}, # Params to init langchain class
            short_term_memory_limit=1024, # Size of short-term memory in tokens 
            embedding_model="AzureOpenAIEmbeddings", # Same langchain class
            embedding_model_params = {...} # Params to init langchain class
            vectorstore="Chroma", # Same langchain class
            vectorstore_params={...} # Params to init langchain class
            tools=[] # Optinal list of tools for agent to use
            contractors=[] # Optional list of other agents to communicate with
            ctx=ctx # Context shared between agents, have common configurations
        )

    @property
    def name(self):
        return "Customer Support Assistant"

    @property
    def description(self):
        return (
            "Assists users with common queries and direct complex issues to the appropriate teams."
        )
```

`name` and `description` properties used to clearly articulate agents identity and functionality. These properties should be descriptive and concise, offering a clear understanding of the agent's purpose and capabilities.

With the `name` and `description` properties defined, your custom agent is now ready to be used as a contractor by other agents. Ensure your agent's functionalities are accessible and well-documented, allowing other agents to leverage its capabilities effectively.


#### Step 3: Interact with Your Agent
Finally, interact with your agent using the custom logic you've implemented.

```python
from uuid import uuid4
from analysta_llm_agents.tools.context import Context

ctx = Context()
custom_agent = CustomAgent(ctx)

user_query = "I have an issue with my recent order."

for message in custom_agent.start(user_query, conversation_id=str(uuid4())):
    print(message)

```

#### Conclusion

By following these steps, you can create a custom agent profile that is well-suited to your specific tasks and workflows. Experiment with different configurations and functionalities to fully leverage the capabilities of the LLM Agent Framework.


### Creating Custom Tools

Enhancing your Customer Support Assistant Agent with custom tools can significantly improve its ability to serve users more efficiently and effectively. Below is a guide on how to develop and integrate a custom tool for handling frequently asked questions (FAQs) and ticket routing, inspired by the given code structure.

### FAQ Retrieval Tool
This tool fetches frequently asked questions and their answers from a predefined knowledge base or document. This enables the agent to quickly provide users with information without the need for manual lookup or intervention.

```python
def fetchFAQs(ctx: Any, category: str = None):
    """
    Retrieves frequently asked questions and their answers from a knowledge base.
    
    Args:
        category (str, optional): The category of FAQs to retrieve. If None, fetches all FAQs.
    """
    # Example URL to your FAQ knowledge base or API endpoint
    faq_url = "https://yourknowledgebase.com/api/faqs"
    
    try:
        # Optionally filter FAQs by category if provided
        if category:
            faq_url += f"?category={category}"

        # Send the GET request to fetch FAQs
        response = requests.get(faq_url, headers={"Content-Type": "application/json"})
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse and return the FAQs from the response
        faqs = response.json()["faqs"]
        return "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch FAQs: {e}")
        return "ERROR: Unable to retrieve FAQs at this time."

```

#### Ticket Routing Tool
This tool analyzes incoming support tickets and routes them to the appropriate department or support tier based on the content of the ticket and predefined routing rules.

```python
def routeTicket(ctx: Any, ticket_description: str):
    """
    Analyzes a support ticket and routes it to the appropriate department or support tier.
    
    Args:
        ctx (Any): The context of the agent, containing shared data and configurations.
        ticket_description (str): The content of the support ticket.
    """
    # Example logic to determine the department based on ticket description
    if "billing" in ticket_description.lower():
        department = "Billing"
    elif "technical" in ticket_description.lower():
        department = "Technical Support"
    else:
        department = "General Inquiries"

    # Log the routing decision
    logger.info(f"Routing ticket to {department}")

    # Placeholder for actual routing logic (e.g., API call to ticketing system)
    # For demonstration, we'll just return the routing decision
    return f"Ticket routed to {department} department."
```

#### Grouping and using in agent as params

Group tools as a list of functions. 

```python
__all__ = [
    fetchFAQs,
    routeTicket
]
```

Import that list in your `agent.py` file and use within Agent init
```python
from .actions import __all__ as tools

class CustomAgent(Agent):
    
    def __init__(self, ctx, **kwargs):
        super.__init__(
            ...
            tools=tools 
            ...
        )

```

This is it. Enjoy custom tools within your agent

## Contributing 🤝

We welcome contributions! Whether it's bug fixes, feature additions, or improvements to the documentation, feel free to fork the repository and submit a pull request.

##  License ⚖️

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements 🙏

Special thanks to all contributors and the open-source community for support and contributions to this project.

Happy Coding! 🚀👾