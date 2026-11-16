# X / Twitter

Local LLMs on a Mac resume chats the slow way: load the whole KV cache from SSD, then start computing.

So I made them overlap. While the GPU runs layer 1, the SSD is already reading layer 2.

Resuming a 6K-token chat on an M4 MacBook:
32-token reply: 0.46 s → 0.29 s (1.6× faster)
Output: bit-identical

Unified memory makes this a great fit for Apple Silicon: SSD bytes land straight in memory the GPU can use. No copy step.

Open source + benchmarks: [link]

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

Same idea as layer-wise loading in LMCache on NVIDIA servers, now on the Mac.

Code, benchmarks and raw data (MIT): [link]

#LocalLLM #AppleSilicon #MLX #LLMInference
