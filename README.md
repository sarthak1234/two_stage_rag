# two_stage_rag
Implements a RAG with multiquery generation followed retrieved document re-ranking using a bert cross encoder.

Two stage rag follows the following steps for generate relevant context

1. Prompt llm to create multiple similar queries (q)
2. for each of the q queries get k similar docs from the data
3. get a set(kXq) docs 
4. Retrieve top k from the total set using bert cross encoder
5. Re-order highest ranked in the in beginning and end and least ranked in middle to solve lost in the middle problem.
6. use these top k for RAG context 
 Ref - https://www.sbert.net/examples/applications/cross-encoder/README.html (BERT  cross encoder)
 Ref - https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630 (Lost in the middle problem)
