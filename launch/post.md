# X: single post (fits 280 characters; attach launch/purgekv_launch.mp4)

Local LLMs on Mac resume chats by loading the whole KV cache from SSD, then computing.

purgekv streams it: the GPU runs layer i while the SSD reads layer i+1.

M4: 0.46s → 0.29s (1.6×), same output. Llama + Qwen3.

Try it: python examples/demo.py
github.com/PoneeshKumar/purgekv

# X: reply under it

cc @awnihannun @ml_explore: layer-streamed KV resume for MLX. v0.1.0 is out, with benchmarks, tests and raw data. Numbers from other Macs very welcome.

# X: thread version (7 posts)

1/ I made local LLMs on Mac resume chats 1.6× faster, with bit-identical output.

The trick: stop waiting for the whole KV cache to load before computing. Stream it layer by layer instead.

How it works 🧵

2/ When you come back to a chat, the model needs its KV cache: the saved attention keys and values for every past token, for every layer.

For a 6K-token chat on Llama 3.2 3B that's ~670 MB, read back from SSD.

3/ Mac LLM servers (oMLX, LM Studio, vllm-mlx) do this in two steps:

1. Load the entire cache from SSD
2. Then run your new message through the model

Time to first token = load + compute. The GPU idles during the load, then the SSD idles during compute.

4/ But layer 5 only needs layer 5's cache, and only when the forward pass reaches it.

So: start computing immediately. I/O threads read layer i+1's KV from SSD while the GPU runs layer i (MLX async_eval).

Time to first token ≈ max(load, compute), not the sum.

5/ Apple Silicon makes this simple. With unified memory, bytes read from SSD land in memory the GPU reads directly.

No host-to-device copy to schedule, which is the stage NVIDIA setups have to pipeline separately.

6/ Results on an M4 MacBook (3B model, 6K-token chat, cold SSD reads, median of 6):

32-token reply: 0.46s → 0.29s (1.6×)
256-token reply: 1.25s → 0.99s (1.3×)

The gain is largest when load ≈ compute: short messages, exactly what agents send.

7/ v0.1.0 is out (MIT). Works with Llama 3.x and Qwen3, with tests and raw data.

Measure it on your own Mac in one command: python examples/demo.py

github.com/PoneeshKumar/purgekv

# LinkedIn

Making local LLMs resume chats faster on Apple Silicon

When you come back to a chat or an agent session, Mac LLM servers reload its saved KV cache from SSD, and only then start computing. The GPU waits for the SSD, then the SSD waits for the GPU.

I built layer-streamed resume for MLX: the model starts on your new message immediately and reads each layer's cache just before that layer needs it. SSD reads and GPU compute run at the same time.

Results on an M4 MacBook (Llama 3.2 3B, 6K-token chat, cold SSD read):
• 32-token message: 0.46 s → 0.29 s (1.6× faster)
• 256-token message: 1.25 s → 0.99 s (1.3× faster)
• Output bit-identical to an uninterrupted run, every time

Short messages are the sweet spot. That's exactly what agents send: tool results and quick replies.

Apple Silicon suits this well. With unified memory, bytes read from SSD go straight into memory the GPU uses, so there's no host-to-device copy to manage.

purgekv v0.1.0 is now out: it supports Llama 3.x and Qwen3, ships with tests and raw benchmark data, and has a one-command demo so you can measure the speedup on your own Mac (python examples/demo.py).

Same idea as layer-wise loading in LMCache on NVIDIA servers, now on the Mac.

Code (MIT): https://github.com/PoneeshKumar/purgekv

#LocalLLM #AppleSilicon #MLX #LLMInference
