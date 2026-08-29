---
title: Programming the Agentic Stack
author: Federico Mora
webpage: federico.morarocha.ca
---

# Programming the Agentic Stack

What does it mean to program an agent? What is really happening under-the-hood?
What can go wrong and what can we do about it?

This fall, I am teaching [CS 846:
FMxAI](https://federico.morarocha.ca/CS846-FMxAI/) at the University of
Waterloo. This blog post is an informal companion to the first three modules of
the course.

## User Interface (Agentic SDK)

Agentic SDKs let you define agents in just a few lines of code. For example,
the code below defines a simple agent called `fib_agent` that has access to one
tool, `fib`, which takes in an integer `n` and returns the `n`-th number in the
Fibonacci sequence. This is a silly example, but we will see that it is
surprisingly illustrative.

```python
@tool
def fib(n: int) -> int:
    """Return the nth Fibonacci number"""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fib_agent = Agent(tools=[fib])
```

Once we have the agent, we can give it tasks. For example, we can ask it for
the 33rd Fibonacci number---and it gets the answer right!

```pycon
>>> answer = fib_agent("What is the thirty-third Fibonacci number?")
>>> print(answer)
The thirty-third Fibonacci number is 3524578
```

## The Mechanics (Inference Engine)

But what in the world just happened? Did the agent use the tool? If so, how?
How did it know that the tool exists? How did it call it? How did it get the
result? What do language models have to do with any of this? What is happening?

Roughly, under-the-hood, we are 

1. converting our agentic program into a prompt following a _template_;
2. converting that prompt into _tokens_;
3. using a probability distribution to _generate_ more tokens; 
4. _recognizing_ tool calls in the tokens;
5. _calling_ those tools;
6. _returning_ the result of the tool call as more tokens; and
7. repeating from step 3 until we generate a special _end_ token.

Many of these steps are language model specific. To make things as concrete as
possible, we will walk through the toy example using Qwen3 syntax and
standards.

### 1. Template

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

### 2. Tokens

Dictionary plus rules (order matters)

### 3. Generate

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

### 4. Recognize

```python
# detokenize (order doesn't matter in this direction)
# parse the json in <tool_call>...</tool_call>
```

If the language model uses slightly different format, you can run into trouble.
For example, today, Claude gave me output like this `(cite
index="9-1">...</cite>` which probably should have rendered as a clickable
citation. I suspect the parser was expecting an opening angle bracket, `<`,
instead of an opening round bracket, `(`, before the word `cite`. Since the
harness did not recognize the citation, the text flowed through to me, the
user, directly.

### 5. Call

### 6. Return

For Qwen, "tool results are treated as special user messages." Models like GLM use an actual `<|observation|>` role for this instead.

```qwen3
<|im_start|>user
<tool_response>
3524578
</tool_response>
<|im_end|>
```

### 7. End

```qwen3
<|im_start|>assistant
<think>
The fib function returned 3524578. That's the answer to give the user.
</think>
The thirty-third Fibonacci number is 3524578
<|im_end|>
```

## Concerns and Interventions

### Token Cost and Context Window Size

Claude Fable 5 tokens cost $10 / MTok for inputs and $50 / MTok for outputs.
Each digit is one token so if I ask this agent for the 10,367,321st Fibonacci
number, for example, the answer, which has 2,166,642 digits, would cost me
$130, if it could fit in the context window. Claude Fable 5 has a 1M token
context window, so this wouldn't work anyway.

Fibonacci is a silly example, but there are plenty of realistic tools that have
the same issue. An agent that reads from a database might bite off more than it
can chew (or that you can afford for it to chew). An agent that uses an
automated theorem prover might not be able to digest the proof it gets back.

#### Solution 1: Prompting

We could ask the LLM politely to not call `fib` with too large of an argument.

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

