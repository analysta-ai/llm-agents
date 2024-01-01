agent_response_format = """{
    "thoughts": {
        "text": "thought and results of analysis",
        "reasoning": "crisp reasoning, preferably in 2-4 bullet points",
        "plan": "short bulleted, list that conveys long-term plan",
        "criticism": "constructive self-criticism",
    },
    
    "command": {
        "type": "agent|command"
        "name": "command or agent name",
        "args": {
            "arg name": "value"
        }
    }
}"""

gk_response_format = """{
  "valid_question": true|false, 
  "result": "short explanation of decision"
}"""