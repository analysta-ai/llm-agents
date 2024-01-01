agent_response_format = """{
    "thoughts": {
        "text": "thought and results of analysis",
        "reasoning": " crisp reasoning, preferably in 2-4 bullet points",
        "plan": "short bulleted, list that conveys long-term plan",
        "criticism": "constructive self-criticism",
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}"""


analyze_prompt = """Having following data as an input, solve the following task:
Task: 
{hypothesis}


Data For Analysis:
{data}

Expected result:
Provide exact answer for a taks, not the way to solve it.
"""
