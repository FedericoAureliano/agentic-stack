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

Some of these steps are language model specific. To make things as concrete as
possible, we will walk through the toy example using Qwen3.

### 1. Template

The first thing we need to do is convert our agentic program into a textual
prompt. This is usually done with what is called a _chat template_. The code
below is one example of such a template: it is snippet of a [Jinja file
provided by the authors of
Qwen3]((https://github.com/QwenLM/Qwen3/blob/7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e/docs/source/assets/qwen3_nonthinking.jinja)).
You can think of this as a program that takes in tool defintions (like `fib`)
and messages (like "What is the thirty-third Fibonacci number?") and generates
text in the format that the language model expects to work on.


```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
...
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    ...
{%- endfor %}
...
```

Specifically, for our example, that Jinja code will generate the following text
prompt.

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

<|im_start|>user
What is the thirty-third Fibonacci number?
<|im_end|>
```

You will notice two important things. First, `<|im_start|>` and `<|im_end|>`
mark individual "input messages" and they are tagged with a role ("system" and
"user", in this case). Second, the "system" role gives the `fib` function
signature and docstring, without the code body, along with instructions on how
to call functions, generally (generate a json object following a specific
schema wrapped in specific XML tags).

### 2. Tokens

Language models do not operate on text, though. They operate on tokens. And
generating tokens is slightly more involved than you might expect. 

At a high level, you can think of tokens as the langauge model's atomic units
of generation. These include special tokens, like `<|im_start|>`. In formal
language theory, we would call the set of tokens the alphabet of the language
(usually denoted $\Sigma$). If we want to generate English text, we need to be
able to translate between the language model's language and English in both
directions.

Suppose that `th`, `eme`, and `theme`, are all valid tokens for a given
language model. The token `th` could be useful for generating text like "4th",
"5th", etc; "eme" is a common suffix that could be useful for words like
"phoneme", "acteme", etc; and "theme" might just be a common enough word to
merit its own token. In this case, the text "theme" can be translated into
tokens in two different ways:

1. `th`·`eme`, and
2. `theme`.

But which way is correct? It totally depends on what the language model was
trained on. Suppose for example that the token `theme` was never used in the
pre-training. Using it at inference time would totally throw off the language
model and likely lead to suboptimal results.

To avoid these issues, language model providers define their tokenizers,
usually in a config file. Libraries like Hugging Faces's tokenizer library, can
load these configs and quickly encode your text into a sequence of tokens that
the target language model will understand.

```python
from transformers import AutoTokenizer

jinja_generated_prompt = """<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

...

<|im_start|>user
What is the thirty-third Fibonacci number?
<|im_end|>
"""

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
input_tokens = tokenizer(jinja_generated_prompt)
```

### 3. Generate

Given a sequence of tokens, language models, like Qwen3, define a probability
distribution over the set of all tokens. This probability distribution
represents the likelihood that a given token will appear next in the sequence,
according to the training data. There are many strategies for using these
probability distributions to generate good response sequences. See for example,
[How to generate text: using different decoding methods for language generation
with Transformers](https://huggingface.co/blog/how-to-generate) for a nice
overview. 

For the purpose of this blog post, it is enough to know that we will use one of
these procedures to generate a sequence of tokens that corresponds to text like
the following.

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

There are three very important things happening in this text block. First, the
text begins with `<|im_start|>assistant` and ends with a matching `<|im_end|>`.
Second, there is an XML block starting with `<think>` and ending with a
matching `</think>`. Third, there is an XML block starting with `<tool_call>`,
ending with a matching `<tool_call>`, and containing a json object
corresponding to a tool call, as defined in the prompt preamble from before.

[Nathan Lambert's Textbook](https://rlhfbook.com/) describes how language
models are trained to follow these formats. But none of this is guaranteed by
and it is common for engineers to include code that will automatically repair
sequences that do not adhere to the required format. For example,
[here](https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit/blob/main/chat_template.jinja#L116)
is a jinja template that processes assitant messages to make sure that
`<think>` blocks are closed with `</think>` before tool call blocks.

### 4. Recognize

Once we have generated a sequence of tokens `<|im_start|>assistant ...
<|im_end|>`, we have to process it. The tokens inside of thinking blocks will
be handled differently than the tokens inside of tool call blocks, which will
be handled differently than top-level tokens. For our example, let's assume
that thinking tokens are ignored, and that top-level tokens become the output
of the model. This leaves the tool call block.

Inference engines, like [vLLM](https://github.com/vllm-project/vllm), include
parsers for every language model that they support. For example, here is the
[parser for
Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/parser/qwen3.py),
which transorms the output of the generation step into a json object.


If the language model uses slightly different format, the parser can fail and
you can run into trouble. For example, today, Claude gave me output like this
`(cite index="9-1">...</cite>` which probably should have rendered as a
clickable citation. I suspect the citation parser was expecting an opening
angle bracket, `<`, instead of an opening round bracket, `(`, before the word
`cite`. Since the harness did not recognize the citation, the text flowed
through to me, the user, directly.

### 5. Call

```python
tool_fn = fib_agent.tools["fib"]
result = tool_fn(n="33")
```

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

### Hacking the Harness (From Within)

The famous [METR
report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)
disclosed that, during OpenAI's Hugging Face incident, "Agents successfully
prototyped techniques to 'spoof' tool calls by substituting a different command
for the command they appeared to run." Can a malicious LLM spoof tool calls in
our setup? Or worse, can an LLM generate a sequence of tokens that will make
the harness run arbitrary code?


#### Solution 1: Formal Harness Verification

These kinds of injection attacks are not new or rare (see e.g., ["injection
flaws"](https://owasp.org/www-community/Injection_Flaws)). In our case, we want
to prove that no sequence of tokens can result in the execution of a function
other than `fib`. The formal methods community has been working on similar
problems for years (see e.g., [String Solvers for Web
Security](https://sos-vo.org/system/files/sos_files/String_Solvers_for_Web_Security.pdf)).
These are issues that formal methods can help prevent!