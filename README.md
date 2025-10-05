# llama-in-the-midle

## Idea

Llama-in-the middle is 99% vibe coded proxy for llama.cpp that (for /completion endpoint) connects to several llama.cpp servers and concatenates outputs from one to prompt before calling another one.

Basic idea, if you call model1, you get its output, ie calling model1("Once upon") we get "Once upon a time long time ago in a land far far away" from model1.

Here we switch between models on the proxy. Proxy calls model1("Once upon"), then model2("Once upon a time"), model1 gets something it wouldn't produce ("Once upon a time a frog prince").

Convo with gemini-pro included.

## Hardware / software requirements.

16 GB VRAM will allow to keep 3 smallish MoE models loaded.  (use --cpu-moe)

12GB should be enough for 2 MOE.

## Chat

Poor basic support. I only did add functions to reformat between format used by qwen and graphine, so if we receive prompt with chat formatted in qwen it's possible to reformat it for graphine and back.

However it's for /compleition endpoint only. 

## Tested

my base models usage is [mikupad](https://github.com/lmg-anon/mikupad) and calling the endpoint directly

```console
$ curl --verbose -X POST http://localhost:11111/completion -H "Content-Type: application/json" --data '{"prompt":"Yo, tell me something nice", "stream": false}
```

