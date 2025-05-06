# LLM vs LLM - Fault Localisation Approach    
Fault localization is a critical and time-consuming step in software
debugging, traditionally requiring extensive tests or instrumenta-
tion. We propose a novel approach in which two large language
models (LLMs) form a self-improving pair (inspired by generative
adversarial setups): one LLM acts as a fault injector, generating
C++ programs with synthetic bugs, and another LLM serves as a
debugger, localizing the buggy code. The fault injector LLM can
dynamically produce an unlimited supply of diverse, realistic faults
beyond existing datasets, while the debugger LLM continuously
fine-tunes on this growing dataset to improve its accuracy. We
implemented a prototype using the Gemma-3-12B model to gener-
ate 5000 buggy programs and fine-tuned a similar 12B-parameter
model via parameter-efficient techniques (PEFT with QLoRA) on
this synthetic corpus. Preliminary results show that the fine-tuned
debugger exhibits modest improvements in pinpointing bug lo-
cations (with training loss decreasing and accuracy improving)
compared to its base state. However, challenges such as limited
computing resources, long training times, and ensuring fault re-
alism hindered full realization of the iterative training loop. We
discuss these challenges, validate our approach against background
literature, and outline future directions including incorporation of
real bug benchmarks (Defects4J, CodeNet), pipeline optimizations,
and a reinforced feedback loop between the LLMs. Our findings
suggest that with adequate resources, an LLM-vs-LLM framework
could significantly advance automated fault localization.
