# Lattice_LSTM
lattice lstm cell implementation with tensorflow

This is the generic implementation of [Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf)

# why lattice?
The traditional name entity regconition task usally RNN cell to be the core unit for designing it's architecture. The common one will be used together with bidirectional architecture. But the performance usually not that great as structure of language is so complex that there are too much possiblility for ambiguity, especially for chinese. 

lattice lstm induce lexicon words for the character inputs so that the whole units can learn the hidden feature between characters data manifold from lexicon data manifold. 
