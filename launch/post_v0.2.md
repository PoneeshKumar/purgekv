Part 2: local AI chats now resume 5.8× faster at 32K tokens.

16GB M4 MacBook, Llama 3.2 3B:
Recompute: 169s
Reload today: 1.80s
purgekv v0.2: 0.31s

The trick: it reads the chat's memory off the SSD while you type.

Same output. Open source:
github.com/PoneeshKumar/purgekv
