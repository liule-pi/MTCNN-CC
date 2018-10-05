#!/bin/bash

# freeze the graph and the weights
python freeze_graph.py --input_graph=./ckpt/pnet/pnet.pbtxt --input_checkpoint=./ckpt/pnet/pnet --output_graph=./ckpt/pnet/pnet_frozen.pb --output_node_names=pnet/conv4-2/BiasAdd,pnet/prob1
python freeze_graph.py --input_graph=./ckpt/rnet/rnet.pbtxt --input_checkpoint=./ckpt/rnet/rnet --output_graph=./ckpt/rnet/rnet_frozen.pb --output_node_names=rnet/conv5-2/conv5-2,rnet/prob1
python freeze_graph.py --input_graph=./ckpt/onet/onet.pbtxt --input_checkpoint=./ckpt/onet/onet --output_graph=./ckpt/onet/onet_frozen.pb --output_node_names=onet/conv6-2/conv6-2,onet/conv6-3/conv6-3,onet/prob1
