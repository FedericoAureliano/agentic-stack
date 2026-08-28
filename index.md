---
title: Programming the Agentic Stack
author: Federico Mora
webpage: federico.morarocha.ca
---

# Programming the Agentic Stack



## User Interface (Agentic SDK)

```python
@tool
def fib(n: int) -> int:
    """Return the nth Fibonacci number"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fib_agent = Agent(tools=[fib])

answer = fib_agent("What is the thirty-third Fibonacci number?")
print(answer) # The thirty-third Fibonacci number is 3524578
```

## The Mechanics (Inference Engine)


### Chat Template

```jinja
{%- if tools %}
  {{- '<|im_start|>system\n' }}
  {%- if messages[0].role == 'system' %}
    {{- messages[0].content + '\n\n' }}
  {%- endif %}
  {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
  {%- for tool in tools %}
    {{- "\n" }}{{- tool | tojson }}
  {%- endfor %}
  {{- "\n</tools>\n\nFor each function call, return a json object..." }}
{%- endif %}
```

```qwen3
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
  "type"    : "function", 
  "function": {
    "name"       : "fib", 
    "description": "Return the nth Fibonacci number", 
    "parameters" : {
      "type"      : "object",
      "properties": {"n": {"type": "integer"}}, 
      "required"  : ["n"]
    }
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>
```

```qwen3
<|im_start|>user
What is the thirty-third Fibonacci number?
<|im_end|>
```

### Tokenizer

Dictionary plus rules (order matters)

### Token Generation

```qwen3
<|im_start|>assistant
<think>
The user wants the 33rd Fibonacci number. Rather than compute it by hand, I'll call the fib function with n=33.
</think>
<tool_call>
{"name": "fib", "arguments": {"n": 33}}
</tool_call>
<|im_end|>
```

### Reasoning and Tool-calls (Parse and Respond)

```python
# detokenize (order doesn't matter in this direction)
# get the json in <tool_call>...</tool_call>
# call the corresponding function
# send back the response
```

For Qwen, "tool results are treated as special user messages." Models like GLM use an actual `<|observation|>` role for this instead.

```qwen3
<|im_start|>user
<tool_response>
3524578
</tool_response>
<|im_end|>
```

### Final Response

```qwen3
<|im_start|>assistant
<think>
The fib function returned 3524578. That's the answer to give the user.
</think>
The thirty-third Fibonacci number is 3524578
<|im_end|>
```

## Concerns and Interventions

### Token Cost and Context Size

#### Solution 1: Prompting

#### Solution 2: Design

#### Solution 3: Monitors (Hooks)

### Tool Interface Errors

#### Solution 1: Prompting

#### Solution 2: Grammar Constraind Decoding

##### Do Tokens and Non-Terminals Align?

### Multiple Tool Calls

Imagine that we ask for two `fib` tool calls, we execute them in parallel, and the results come back out of order. Will we give the user the right response? 

#### Solution 1: Tool Call IDs

Qwen doesn't support tool call IDs, but other models do.

#### Solution 2: Pure Functions

### End-to-End Correctness

