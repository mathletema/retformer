python train_retnet.py --dmodel=128 --dffn=256 --nlayer=2 --nheads=2 --chunksize=512 --batchsize=4 --lr1=0.01 --lr2=0.001 --beta1=0.8 --beta2=0.9 --weightdecay=0.01 --warmupsteps=50 --dropprob=0.05 --numepochs=2 --printevery=10 --twoorthree=2 --isdistributed=0 \
    --savenamefinal="finalpineapple" --savenamebest="bestpineapple"

python -m torch.distributed.launch --nproc_per_node=6 train_retnet.py --dmodel=128 --dffn=256 --nlayer=2 --nheads=2 --chunksize=512 --batchsize=4 --lr1=0.01 --lr2=0.001 --beta1=0.8 --beta2=0.9 --weightdecay=0.01 --warmupsteps=50 --dropprob=0.05 --numepochs=2 --printevery=10 --twoorthree=2 --isdistributed=0 \
    --savenamefinal="finalpineapple" --savenamebest="bestpineapple"
