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
merge_dir="../output"
model="gpt-4o"
num_templates_for_task=360

output_dirs_str=$(IFS=' '; echo "${output_dirs[*]}")
length=${#api_keys[@]}

# add '&' to run commands in parallel
for ((i=0; i<$length; i++))
do
  python data_synthesis.py \
    --api_key ${api_keys[i]} \
    --base_url ${base_urls[i]} \
    --output_dir ${output_dirs[i]} \
    --model ${model} \
    --num_templates_for_task ${num_templates_for_task} \
    --seed ${i} & 
done

wait 

# merge the results 
echo "Merging the results..."
python merge_templates.py --input_dirs ${output_dirs_str} --output_dir ${merge_dir}
# split templates for each task 
echo "Splitting the templates..."
python split_templates.py --output_dirs ${merge_dir} --model ${model}

echo "Done!"
exit 0