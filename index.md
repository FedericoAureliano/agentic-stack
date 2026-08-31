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

## User Interface

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

## The Mechanics

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
7. repeating from step 3 until we reach a stopping condition.

This is known as the agentic loop. Some of the steps are language model
specific. To make things as concrete as possible, we will walk through the toy
example using Qwen3.

### 1. Template

The first thing we need to do is convert our agentic program into a textual
prompt. This is usually done with what is called a _chat template_. The code
below is one example of such a template: it is a simplified snippet---with
added comments---of the [chat template for
Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/chat_template.jinja).
You can think of this as a program that takes in tool definitions (like `fib`)
and messages (like "What is the thirty-third Fibonacci number?") and generates
text in the format that the language model expects to work on.


```jinja
{# Emit the system message: pass the caller's through, or synthesize a
   default one when tools were provided but no system message was given #}
{%- if system_message is defined %}
    {{- "<|im_start|>system\n" + system_message }}
{%- else %}
    {%- if tools is iterable and tools | length > 0 %}
        {{- "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
    {%- endif %}
{%- endif %}

{# List every tool as an XML <function> block inside <tools>...</tools> #}
{%- if tools is iterable and tools | length > 0 %}
    {{- "\n\n# Tools\n\nYou have access to the following functions:\n\n" }}
    {{- "<tools>" }}
    {%- for tool in tools %}
        {%- if tool.function is defined %}
            {%- set tool = tool.function %}
        {%- endif %}
        {{- "\n<function>\n<name>" ~ tool.name ~ "</name>" }}
        {%- if tool.description is defined %}
            {{- '\n<description>' ~ (tool.description | trim) ~ '</description>' }}
        {%- endif %}
        {{- '\n<parameters>' }}
        ...
        {{- '\n</parameters>' }}
        {{- '\n</function>' }}
    {%- endfor %}
    {{- "\n</tools>" }}

    {# Tell the model exactly how a tool call must be formatted #}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n...\n</function>\n</tool_call>\n\n...' }}
{%- endif %}
...

{# Walk the conversation, emitting one <|im_start|>...<|im_end|> block per message #}
{%- for message in loop_messages %}
    {# An assistant message with tool calls becomes one <tool_call> per call,
       nesting the function name and arguments as <function=NAME><parameter=NAME> #}
    {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
        {{- '<|im_start|>' + message.role }}
        ...
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
            {%- for args_name, args_value in tool_call.arguments|items %}
                {{- '<parameter=' + args_name + '>\n' }}
                ...
                {{- '\n</parameter>\n' }}
            {%- endfor %}
            {{- '</function>\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}

    {# Otherwise, pass the message through as-is #}
    {%- elif message.role == "user" or message.role == "system" or message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    ...
{%- endfor %}
...
```

Specifically, for our example, that Jinja code will generate the following text
prompt.

```qwen3
<|im_start|>system
You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.

# Tools

You have access to the following functions:

<tools>
<function>
<name>fib</name>
<description>Return the nth Fibonacci number</description>
<parameters>
<parameter>
<name>n</name>
<type>integer</type>
</parameter>
<required>["n"]</required>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
<|im_end|>

<|im_start|>user
What is the thirty-third Fibonacci number?
<|im_end|>
```

You will notice two important things. First, `<|im_start|>` and `<|im_end|>`
mark individual "input messages" and they are tagged with a role ("system" and
"user", in this case). Second, the "system" role gives the `fib` function
signature and docstring, without the code body, along with instructions on how
to call functions, generally (wrap the call in `<tool_call>` and
`<function=example_function_name>` tags, with the argument in its own
`<parameter=example_parameter_1>` tag).

### 2. Tokens

Language models do not operate on text, though. They operate on tokens. And
generating tokens is slightly more involved than you might expect.

At a high level, you can think of tokens as the language model's atomic units
of generation. These include special tokens, like `<|im_start|>`. In formal
language theory, we would call the set of tokens the alphabet of the language
(usually denoted $\Sigma$). If we want to generate English text, we need to be
able to translate between the language model's language and English in both
directions.

Suppose that `th`, `eme`, and `theme` are all valid tokens for a given
language model. The token `th` could be useful for generating text like "4th",
"5th", etc; "eme" is a common suffix that could be useful for words like
"phoneme", "acteme", etc; and "theme" might just be a common enough word to
merit its own token. In this case, the text "theme" can be translated into
tokens in two different ways:

1. `th`·`eme`, and
2. `theme`.

<div class="bug-banner"><span class="bug-emoji" title="known rough edge" aria-hidden="true">🐛</span></div>
But which way is correct? It totally depends on what the language model was
trained on. Suppose for example that the token `theme` was never used in the
pre-training. Using it at inference time would totally throw off the language
model and likely lead to suboptimal results.

To avoid these issues, language model providers define their tokenizers,
usually in a config file. Libraries like Hugging Face's tokenizer library can
load these configs and quickly encode your text into a sequence of tokens that
the target language model will understand.

```python
from transformers import AutoTokenizer

jinja_generated_prompt = """<|im_start|>system
You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.

# Tools

You have access to the following functions:

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
<function=fib>
<parameter=n>
33
</parameter>
</function>
</tool_call>
<|im_end|>
```

There are three very important things happening in this text block. First, the
text begins with `<|im_start|>assistant` and ends with a matching `<|im_end|>`.
Second, there is an XML block starting with `<think>` and ending with a
matching `</think>`. Third, there is an XML block starting with `<tool_call>`,
ending with a matching `</tool_call>`, and containing a nested `<function=fib>`
block whose `<parameter=n>` tag carries the argument, as defined in the prompt
preamble from before.

<div class="bug-banner"><span class="bug-emoji" title="known rough edge" aria-hidden="true">🐛</span></div>
[Nathan Lambert's Textbook](https://rlhfbook.com/) describes how language
models are trained to follow these formats. But none of this is guaranteed by
and it is common for engineers to include code that will automatically repair
sequences that do not adhere to the required format. For example,
[here](https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit/blob/main/chat_template.jinja#L116)
is a jinja template that processes assistant messages to make sure that
`<think>` blocks are closed with `</think>` before tool call blocks.

### 4. Recognize

Once we have generated a sequence of tokens `<|im_start|>assistant ...
<|im_end|>`, we have to process it. The tokens inside of thinking blocks will
be handled differently than the tokens inside of tool call blocks, which will
be handled differently than top-level tokens. Inference engines, like
[vLLM](https://github.com/vllm-project/vllm), include parsers for every
language model that they support. For example, here is the [parser for
Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/parser/qwen3.py),
which transforms the output of the generation step into a JSON object. This is
what we mean by "recognize" the tool calls, thinking steps, and responses: find
them and package them up for the next step.

```json
{
  "role": "assistant",
  "content": null,
  "reasoning_content": "The user wants the 33rd Fibonacci number. Rather than compute it by hand, I'll call the fib function with n=33.",
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "fib",
        "arguments": "{\"n\": 33}"
      }
    }
  ]
}
```

<div class="bug-banner"><span class="bug-emoji" title="known rough edge"
aria-hidden="true">🐛</span></div> If the language model uses slightly
different format, the parser can fail and you can run into trouble. For
example, today, Claude gave me output like this `(cite index="9-1">...</cite>`
which probably should have rendered as a clickable citation. I suspect the
citation parser was expecting an opening angle bracket, `<`, instead of an
opening round bracket, `(`, before the word `cite`. Since their parser did not
recognize the citation, the text flowed through to me, the user, directly. You
can find another similar example in the vLLM project, where [a failure in the
parser resulted in silently dropping tool
calls](https://github.com/vllm-project/vllm/issues/39056).

### 5. Call

Once we have the response in JSON format, we can process it and actually make
the tool calls that the language model requested. The language model cannot do
anything directly. The agent harness, the inference runtime, and the serving
engine are its arms, feet, and heart. The following code shows how we could
implement the tool calls themselves in an agentic harness, assuming the parsing
results are in a variable called `response`.

```python
import json

results = {}
for tool_call in response["tool_calls"]:
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    tool_fn = fib_agent.tools[name]
    results[tool_call["id"]] = tool_fn(**args)
```

### 6. Return

For Qwen3, "tool results are treated as special user messages." Models like GLM
use an actual `<|observation|>` role for this instead. Either way, the chat
template handles the encoding so that we don't have to worry about the details.
For Qwen3, the result of our tool call can be sent back to the language model
as the following text (appropriately tokenized by the tokenizer, as before).

```qwen3
<|im_start|>user
<tool_response>
3524578
</tool_response>
<|im_end|>
```

At this point, we return to step 3 to generate more tokens using the underlying
language model.

### 7. End

Eventually, we will generate a sequence of tokens that has no tool calls, like
the following.

```qwen3
<|im_start|>assistant
<think>
The fib function returned 3524578. That's the answer to give the user.
</think>
The thirty-third Fibonacci number is 3524578
<|im_end|>
```

If we want, we could send a message to the language model telling it to think
some more or that it needs to continue for some other reason. If the language
model asks the user a question, perhaps we can identify that and start an
interaction with an end-user. Otherwise, these tool-call-free messages can be
interpreted as the end of the agentic loop and we can return the content of the
JSON message as the final response.

```json
{
  "role": "assistant",
  "content": "The thirty-third Fibonacci number is 3524578",
  "reasoning_content": "The fib function returned 3524578. That's the answer to give the user.",
  "tool_calls": []
}
```

In short, that is how we get the following behaviour that we started with.

```pycon
>>> answer = fib_agent("What is the thirty-third Fibonacci number?")
>>> print(answer)
The thirty-third Fibonacci number is 3524578
```

## Concerns and Interventions

A million things can and do go wrong during this process. In the text above, I
noted three example bugs. One related to tokenizing, one related to instruction
following formats, and one related to parsing. In this section, I will quickly
go over a few more pressing concerns and possible ways to deal with them.

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

We could ask the language model politely to not call `fib` with too large of an
argument. That would probably work most of the time.

We could design the `fib` function so that it rejects large values of `n`.

```python
def fib(n: int) -> int:
    """Return the nth Fibonacci number"""
    if n >= 1000:
        return -1
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

We could also instrument the harness or inference engine to avoid such cases,
generally. For example, we could add a code hook that intercepts all messages to
the language model and, if they are too big, redirects them to a file or just
outright replaces them with a warning message.

### Tool Interface Errors

The one argument to our `fib` function is very simple: it is just an integer
number. But many other tools have much more complicated types. For example, in
[one of our recent papers](https://arxiv.org/pdf/2608.05493), we found that 23%
of services expose a string parameter that must conform to a domain-specific
language and that for one AWS domain-specific language, language models
frequently make syntactic mistakes. We can give language models tools, but they
might just struggle to use them correctly.

There is a growing body of work that addresses these issues with prompting,
constrained decoding, and other kinds of techniques. We will survey this area
on Sep 29 in the [FMxAI course](https://federico.morarocha.ca/CS846-FMxAI/).

### Concurrent Tool Calls

Imagine that we ask for two `fib` tool calls, we execute them in parallel, and
the results come back out of order. Will we give the user the right response?
Or what if these tools have side-effects and interfere with each other? Tool
calling can lead to many concurrency bugs which agent harnesses and SDKs must
deal with, but very few do.

### Hacking the Harness (From Within)

The famous [METR
report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)
disclosed that, during OpenAI's Hugging Face incident, "Agents successfully
prototyped techniques to 'spoof' tool calls by substituting a different command
for the command they appeared to run." Can a malicious language model spoof
tool calls in our setup? Or worse, can a language model generate a sequence of
tokens that will make the harness run arbitrary code? That might sound crazy,
but here is a [pull request in the vLLM
project](https://github.com/vllm-project/vllm/pull/21396#discussion_r2223397938)
that attempted to add a direct Python `eval` call in the parser for tool call
parameters.

More sophisticated code injection attacks from within are likely possible.
Luckily, these kinds of attacks are not new (see e.g., ["injection
flaws"](https://owasp.org/www-community/Injection_Flaws)) and we have tools to
defend against them (see e.g., [String Solvers for Web
Security](https://sos-vo.org/system/files/sos_files/String_Solvers_for_Web_Security.pdf)).
These are issues that formal methods can help prevent!