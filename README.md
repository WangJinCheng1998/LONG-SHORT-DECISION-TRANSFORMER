# Long-Short Decision Transformer: Bridging Global and Local Dependencies for Generalized Decision-Making

This repository provides the official implementation of **Long-Short Decision Transformer (LSDT): Bridging Global and Local Dependencies for Generalized Decision-Making**, which introduces a two-branch Transformer architecture combining self-attention and convolution for reinforcement learning (RL) tasks with both Markovian and non-Markovian properties. 

## Overview

<table>
  <tr>
    <td><img src="Overall_structure.jpg" alt="Image 1" width="350"/></td>
    <td><img src="Detailed_LSDT.jpg" alt="Image 2" width="550"/></td>

  </tr>
</table>

## Dataset Preparation
To use our goal-state concatenation method, you first need to download and preprocess the dataset.

```Dowload_normal_datset
python3 data/download_d4rl_datasets_nogoal.py
```
```Dowload_goal-state concatenation_datset
python3 data/Data_with_Goal_download.py
```

## For training

```Here is an example command for training:
		
		python3 scripts/train.py --env maze2d --dataset large --device cuda --context_len 30 --log_dir maze2d_large 
```


If you want to implement the goal state concatenation, please add " --goalconcate " in the command.  

## For evaluation
```Here is an example command for testing:
	
		python3 scripts/test_o.py --env hopper --dataset medium-expert  --num_eval_ep 10  --chk_pt_name 1_hopper_medium_expert_best.pt --chk_pt_dir /home/usr/LSDT_code/Hopper_medium  --context_len 10 --convdim 96 --render  
```
If you want to implement the goal state concatenation, please add " --goalconcate " in the command.  
The command **convdim** is used to control the dimension ratio which is computed by convdim/(total hidden dimension). 

## Acknowledgements
Our code is based on the implementation of [Decision Convformer](https://github.com/beanie00/Decision-ConvFormer),[Fair](https://github.com/facebookresearch/fairseq/tree/main/fairseq) and [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer).
