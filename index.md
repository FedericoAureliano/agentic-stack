---
title: Programming the Agentic Stack
author: Federico Mora
webpage: federico.morarocha.ca
---

# Programming the Agentic Stack

```python
@tool
def fib(n: int) -> int:
    """Compute the nth Fibonacci number and return the result"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fib_agent = Agent(tools=[fib])
```

```pycon
>>> answer = fib_agent("What is the thirty-third Fibonacci number?")
>>> print(answer)
The thirty-third Fibonacci number is 3524578
```

```qwen3
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "fib", "description": "Compute the nth Fibonacci number and return the result", "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "The index of the Fibonacci number to compute"}}, "required": ["n"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the thirty-third Fibonacci number?<|im_end|>
<|im_start|>assistant
<think>
The user wants the 33rd Fibonacci number. Rather than compute it by hand, I'll call the fib function with n=33.
</think>

<tool_call>
{"name": "fib", "arguments": {"n": 33}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
3524578
</tool_response><|im_end|>
<|im_start|>assistant
<think>
The fib function returned 3524578. That's the answer to give the user.
</think>

The thirty-third Fibonacci number is 3524578<|im_end|>
```