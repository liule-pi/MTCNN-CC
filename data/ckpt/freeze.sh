#!/bin/bash

# freeze the graph and the weights
python freeze_graph.py --input_graph=./pnet/pnet.pbtxt --input_checkpoint=./pnet/pnet --output_graph=./pnet/pnet_frozen.pb --output_node_names=pnet/conv4-2/BiasAdd,pnet/prob1
python freeze_graph.py --input_graph=./rnet/rnet.pbtxt --input_checkpoint=./rnet/rnet --output_graph=./rnet/rnet_frozen.pb --output_node_names=rnet/conv5-2/conv5-2,rnet/prob1
python freeze_graph.py --input_graph=./onet/onet.pbtxt --input_checkpoint=./onet/onet --output_graph=./onet/onet_frozen.pb --output_node_names=onet/conv6-2/conv6-2,onet/conv6-3/conv6-3,onet/prob1
