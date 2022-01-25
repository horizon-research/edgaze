#!/bin/bash

OUTPUT_DIR="output_eye_net_m"

# uncomment this if you use eye_net
# MODEL_WEIGHTS="model_weights/eye_net.pkl"
# MODEL_NAME="eye_net"

# uncomment this if you use pruned_eye_net
# MODEL_WEIGHTS="model_weights/pruned_eye_net.pkl"
# MODEL_NAME="pruned_eye_net"

# uncomment this if you use eye_net_m
MODEL_WEIGHTS="model_weights/eye_net_m.pkl"
MODEL_NAME="eye_net_m"

BBOX_WEIGHTS_PATH="model_weights/G030_c32_best.pth"

# THRESHOLDS=("001" "005" "010" "050" "100")
THRESHOLDS=("001")

for TH in "${THRESHOLDS[@]}"
do
	for num in {0..1}
	do
		python main.py --sequence ${num}  \
		 	--mode filter \
		 	--density_threshold 0.${TH} \
		 	--dataset openEDS \
		 	--model ${MODEL_NAME} \
		 	--pytorch_model_path ${MODEL_WEIGHTS} \
		 	--bbox_model_path ${BBOX_WEIGHTS_PATH} \
		 	--output_path ${OUTPUT_DIR} \
		 	--suffix _${TH} \
		 	--preview
		 	# --disable_edge_info \ 
		 	# --use_sift
	done
done
