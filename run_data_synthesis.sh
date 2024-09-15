#!/bin/bash

api_keys=("api_key_1"
          "api_key_2"
          "api_key_3"
          "api_key_4"
          "api_key_5"
          "api_key_6"
          "api_key_7"
          "api_key_8"
          "api_key_9"
          "api_key_10")
base_urls=("base_url_1"
           "base_url_2"
           "base_url_3"
           "base_url_4"
           "base_url_5"
           "base_url_6"
           "base_url_7"
           "base_url_8"
           "base_url_9"
           "base_url_10")
output_dirs=("../output_1"
             "../output_2"
             "../output_3"
             "../output_4"
             "../output_5"
             "../output_6"
             "../output_7"
             "../output_8"
             "../output_9"
             "../output_10")
world_knowledge_output_dir="../world_knowledge"
merge_dir="../output"
model="gpt-4o"
num_templates_for_task=360

output_dirs_str=$(IFS=' '; echo "${output_dirs[*]}")
length=${#api_keys[@]}

# synthesis the world knowledge 
echo "Stage 1: synthesize the world knowledge"
python data_synthesis.py \
    --api_key ${api_keys[0]} \
    --base_url ${base_urls[0]} \
    --output_dir ${world_knowledge_output_dir} \
    --model ${model} \
    --knowledge_dir "" \
    --only_synthesize_world_knowledge "1"

# add '&' to run commands in parallel
echo "Stage 2: synthesize instruction-response templates"
for ((i=0; i<$length; i++))
do
  python data_synthesis.py \
    --api_key ${api_keys[i]} \
    --base_url ${base_urls[i]} \
    --output_dir ${output_dirs[i]} \
    --model ${model} \
    --num_templates_for_task ${num_templates_for_task} \
    --knowledge_dir ${world_knowledge_output_dir} \
    --only_synthesize_world_knowledge "0" \
    --seed ${i} & 
done

wait 

# merge the results 
echo "Stage 3: merge the templates"
python merge_templates.py --input_dirs ${output_dirs_str} --output_dir ${merge_dir}
# split templates for each task 
echo "Stage 4: further filter and split the templates"
python split_templates.py --output_dirs ${merge_dir} --model ${model}

echo "Done!"
exit 0
