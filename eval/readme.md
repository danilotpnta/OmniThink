In `Knowledge_Density.py`, we use the `factscore` library to decompose atomic knowledge. Please refer to [FActScore](https://github.com/shmsw25/FActScore) for specific API configurations.

After decomposing atomic facts, we use the function `deduplicate_atomic_facts` to remove duplicate atomic knowledge. 
Because `factscore` supports a limited set of older LLMs, we additionally use an LLM outside of `factscore` for deduplication purposes. 
For API key configuration, please see [OmniThink](https://github.com/zjunlp/OmniThink).