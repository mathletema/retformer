This repository contains the code for our final project for NLP 6.8610 called "RetFormers: Hybrid Attention-Retention Mechanisms for Faster Inference". 

### Large-scale experiments
We can run DDP training using the following command or a variant thereof. `binaryvector` represents whether each layer is attention, 0, or retention, 1.
`
torchrun --nproc_per_node=4 train_mixed_retnet_transformer.py --dffn=3072 --chunksize=128 --batchsize=24 --lr1=0.001 --lr2=0.0001 --numepochs=40 --printevery=10000 --isdistributed=1 --savenamebest=typeAbest --savenamefinal=typeAfinal --project=mixedtransformer2 --binaryvector=000000000001
`


### Small-scale experiments and flop measurement
Run the hybrid_retformer.ipynb notebook, and to measure different inference latencies, we can change the forward function in the `Retformer` module to be `forward_efficient_typeA`, `forward_efficient_typeB`, or `forward_efficient_typeC`.