#!/usr/bin/env bash

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")

in="models"
out="pti_out"

for model in ${models[@]}

do

    for i in 0 1

    do 
        # perform the pti and save w
        python projector_withseg.py --outdir=${out} --target_img=dataset/3DDFA_img_korean --network ${in}/${model} --idx ${i} --shapes True --num-steps 2000 --num-steps-pti 1000
        # generate .mp4 before finetune
        python gen_videos_proj_withseg.py --output=${out}/${model}/${i}/PTI_render/pre.mp4 --latent=${out}/${model}/${i}/projected_w.npz --trunc 0.7 --network ${in}/${model} --cfg Head --shapes True --w-frames 960
        # generate .mp4 after finetune
        python gen_videos_proj_withseg.py --output=${out}/${model}/${i}/PTI_render/post.mp4 --latent=${out}/${model}/${i}/projected_w.npz --trunc 0.7 --network ${out}/${model}/${i}/fintuned_generator.pkl --cfg Head --shapes True --w-frames 960


    done

done
