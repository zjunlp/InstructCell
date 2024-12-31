> [!NOTE]
> The code leveraging xFinder for answer extraction is adapted from the [official implementation](https://github.com/IAAR-Shanghai/xFinder/tree/main).

## :zap: Quick Start
1. **Ensure Compatibility**: Ensure you have Python 3.10.0+.
2. **Prepare QA pairs & LLM Outputs**: Prepare the LLM outputs that you want to evaluate. 
   - provide a `.json` file including original question, key answer type (alphabet / short_text / categorical_label / math), LLM output, standard answer range.
3. **Deploy the xFinder Model**: Choose between two models for deployment, [xFinder-qwen1505](https://huggingface.co/IAAR-Shanghai/xFinder-qwen1505) or [xFinder-llama38it](https://huggingface.co/IAAR-Shanghai/xFinder-llama38it).
4. **Finish Configuration**: Compile the above details into a configuration file. For configuration details, see [`xfinder_config.yaml`](xfinder_config.yaml).

After setting up the configuration file, you can proceed with the evaluation:

**Installation**
```sh
conda create -n xfinder_env python=3.11 -y
conda activate xfinder_env
pip install -r requirements.txt
```

**Evaluation with xFinder**
```sh
python eval.py
```

## :memo: Citation
```
@article{xFinder,
      title={xFinder: Robust and Pinpoint Answer Extraction for Large Language Models}, 
      author={Qingchen Yu and Zifan Zheng and Shichao Song and Zhiyu Li and Feiyu Xiong and Bo Tang and Ding Chen},
      journal={arXiv preprint arXiv:2405.11874},
      year={2024},
}
```
