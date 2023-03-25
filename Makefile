test:
	echo tes\
	ting

train_db_testing:
	./docker_run.sh python train_DB_StyleGAN2.py \
	--cfg="auto" \
	--snap=20 \
	--kimg=200 \
	--data="DATASETS/Testing/embedded/train/images" \
	--resume="ffhq256" \
	--gpus=1 \
	--mirror=1 \
	--outdir="IJCB_EXPERIMENTS/DB_SG2/testing" \

train_db:
	./docker_run.sh python train_DB_StyleGAN2.py \
	--cfg="auto" \
	--snap=20 \
	--data="DATASETS/CelebAHQ/train/images" \
	--resume="ffhq256" \
	--gpus=1 \
	--mirror=1 \
	--outdir="IJCB_EXPERIMENTS/DB_SG2/CelebAHQ++" \

generate_images_testing:
	./docker_run.sh python make_training_data_DB_SG2.py \
	--num_sample=100
	--exp="IJCB_EXPERIMENTS/interpreter/testing/generate.json" \
	--sv_path="IJCB_EXPERIMENTS/interpreter/testing"

interpreter_train_testing:
	./docker_run.sh python train_interpreter_DB_SG2.py \
	--exp "IJCB_EXPERIMENTS/interpreter/testing/train_datagan.json"

interpreter_generate_testing:
	./docker_run.sh python train_interpreter_DB_SG2.py \
	--generate_data True \
	--num_sample=50 \
	--exp "IJCB_EXPERIMENTS/interpreter/testing/train_datagan.json" \
	--resume "IJCB_EXPERIMENTS/interpreter/testing" 