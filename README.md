# Consprompt: Exploiting Contrastive Samples for Few-Shot Prompt Learning 
The prompt has become an effective linguistic tool for utilizing
pre-trained language models. However, in few-shot scenarios, subtle changes in the prompt’s design always make the
result widely different, and the prompt learning methods
also easy to overfit the limited samples. To alleviate
this, we explore utilizing suitable contrastive samples and
multi-degree contrastive learning methods to improve the robustness of the prompt’s representation. Therefore, the proposed
Consprompt, combined with the prompt encoding network, contrastive sampling modules, and contrastive scoring modules,
is introduced to realize differential contrastive learning.

Data example can see in the each zip file 16-100 21-100....zip each contains their sbert embedding
we user the standford Sbert Embedding
![image](https://github.com/Nagin-Kim/cosprompt/assets/24890015/a6e64667-882c-4446-9c27-83daffb4a532)
If you use our idea, you can cite our paper.

{J. Weng, Y. Deng, D. Li, H. You, Y. Hu and H. Huang, "Consprompt: Exploiting Contrastive Samples for Few-Shot Prompt Learning," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 6835-6839, doi: 10.1109/ICASSP48485.2024.10448403. keywords: {Self-supervised learning;Signal processing;Linguistics;Controllability;Robustness;Encoding;Acoustics;Prompt learning;Pre-trained language model;contrastive learning;few-shot learning}


___________________________________________________________________________________________________________

### our baseline use the code in :https://gitcode.com/princeton-nlp/LM-BFF
