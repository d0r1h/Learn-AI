__Output from RAG using different methods__

__Q. why does llm hallucinate ?__ 

We have used this [blog](https://lilianweng.github.io/posts/2024-07-07-hallucination/) as context or source. 

* __Using just RAG system without any optimization__


     Based solely on the provided context, LLMs hallucinate for several reasons:

    1.  **Pre-training Data Issues:** The enormous pre-training corpus (often crawled from the public internet) contains out-of-date, missing, or incorrect information. The model can incorrectly memorize this flawed information by maximizing log-likelihood.

    2.  **Fine-tuning New Knowledge:** Introducing new knowledge during fine-tuning (like supervised fine-tuning or RLHF) can encourage hallucinations. Research (Gekhman et al. 2024) found that:
        *   LLMs learn new knowledge from fine-tuning examples slower than they learn examples consistent with their pre-existing knowledge.
        *   Once these new-knowledge examples are eventually learned, they increase the model's tendency to hallucinate.




* __Using Query Translation methods (Multi Query)__

    Based on the context provided, LLMs hallucinate primarily due to issues with their **pre-training data** and challenges in **learning new knowledge during fine-tuning**.

    The key causes outlined are:

    1.  **Pre-training Data Issues**: The massive datasets used for pre-training (often crawled from the public internet) inherently contain **out-of-date, missing, or incorrect information**. Since the model learns by maximizing the likelihood of this data, it can incorrectly memorize and reproduce these inaccuracies, leading to fabricated or unfaithful outputs.

    2.  **Fine-Tuning on New Knowledge**: Introducing new information via supervised fine-tuning can be problematic. Research (Gekhman et al., 2024) found that:
         *   Models learn new knowledge that contradicts their pre-existing knowledge much **slower** than they learn consistent information.
        *   Once the model eventually learns these new, contradictory examples, it **increases its tendency to hallucinate**.
        *   The best model performance is achieved when it learns most of the known information but only a few of the new, unknown facts.

        In summary, hallucinations arise because the model's foundational knowledge base (pre-training data) is imperfect, and the process of updating that knowledge (fine-tuning) can destabilize the

