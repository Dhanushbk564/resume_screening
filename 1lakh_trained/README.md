---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:100000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Cybersecurity Analysts protect computer systems and networks from
    cyber threats. They monitor security systems, investigate breaches, and implement
    security measures to safeguard data.
  sentences:
  - Cybersecurity Security assessments Intrusion detection Security tools (e.g., SIEM)
    Incident response Vulnerability scanning
  - Legal research Document drafting Communication skills Legal software proficiency
    Attention to detail
  - Quality assurance Quality control Process improvement Audit procedures Compliance
    standards Root cause analysis
- source_sentence: Assist in the preparation of litigation cases, manage electronic
    discovery, and handle technical aspects of trial preparation.
  sentences:
  - E-discovery tools and processes Document review and production Litigation support
    software (e.g., Relativity) Data analysis Case management Communication with legal
    teams Technical proficiency Attention to detail Problem-solving Legal research
    skills
  - Account management Client relations Marketing strategies Campaign optimization
    Data analysis Communication skills
  - Database management systems (e.g., MySQL, Oracle, SQL Server) Data security Database
    tuning and optimization Backup and recovery
- source_sentence: Inventory Control Specialists manage and optimize inventory levels
    in a business. They track stock, monitor demand, and implement inventory control
    procedures to ensure efficient operations and minimize carrying costs.
  sentences:
  - Network security Cybersecurity Intrusion detection Security analysis Firewall
    management
  - Inventory management Stock control Demand forecasting Inventory tracking systems
    Supply chain coordination Problem-solving skills
  - Cloud systems engineering Cloud infrastructure (e.g., AWS, Azure) DevOps practices
    Automation Security in the cloud Disaster recovery Scalability
- source_sentence: A Strategic Partnerships Manager identifies and develops strategic
    partnerships and alliances that benefit the organization, fostering collaborations
    and driving mutual success.
  sentences:
  - Aerospace engineering CAD software (e.g., AutoCAD) Aerodynamics Structural analysis
    Aircraft systems Safety regulations
  - User-centered design principles UX/UI design tools (e.g., Sketch, Adobe XD) Wireframing
    and prototyping Usability testing and user research Information architecture and
    user flows
  - Partnership development Negotiation and collaboration Business development
- source_sentence: Creative Copywriters craft engaging and persuasive copy for marketing
    materials, advertisements, and content marketing campaigns. They use their writing
    skills to captivate audiences and convey brand messages effectively.
  sentences:
  - Research methodology Data analysis Psychological studies Writing research reports
    Critical thinking
  - Creative writing Copywriting Advertising copy Content creation Brand storytelling
    Marketing campaigns Proofreading and editing
  - Creative writing Copywriting Advertising copy Content creation Brand storytelling
    Marketing campaigns Proofreading and editing
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Creative Copywriters craft engaging and persuasive copy for marketing materials, advertisements, and content marketing campaigns. They use their writing skills to captivate audiences and convey brand messages effectively.',
    'Creative writing Copywriting Advertising copy Content creation Brand storytelling Marketing campaigns Proofreading and editing',
    'Creative writing Copywriting Advertising copy Content creation Brand storytelling Marketing campaigns Proofreading and editing',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 100,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            |
  | details | <ul><li>min: 14 tokens</li><li>mean: 32.71 tokens</li><li>max: 75 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 23.54 tokens</li><li>max: 79 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                      | sentence_1                                                                                                    |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
  | <code>Product Demonstrators showcase products to potential customers. They explain features, answer questions, and help customers make informed purchase decisions.</code>                                                                                                      | <code>Product knowledge Demonstrating skills Sales techniques Customer engagement Communication skills</code> |
  | <code>A Mergers and Acquisitions Advisor assists in evaluating potential mergers and acquisitions, conducting due diligence, and providing strategic recommendations for corporate transactions.</code>                                                                         | <code>Mergers and acquisitions (M&A) expertise Due diligence Valuation techniques</code>                      |
  | <code>Investment Analysts analyze financial data and market trends to make informed investment recommendations. They research and assess investment opportunities, create financial models, and provide insights to guide investment decisions and portfolio management.</code> | <code>Financial analysis Investment evaluation Portfolio management Data analysis</code>                      |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.08   | 500   | 0.0787        |
| 0.16   | 1000  | 0.047         |
| 0.24   | 1500  | 0.0427        |
| 0.32   | 2000  | 0.0379        |
| 0.4    | 2500  | 0.039         |
| 0.48   | 3000  | 0.0394        |
| 0.56   | 3500  | 0.0375        |
| 0.64   | 4000  | 0.0362        |
| 0.72   | 4500  | 0.039         |
| 0.8    | 5000  | 0.0384        |
| 0.88   | 5500  | 0.0351        |
| 0.96   | 6000  | 0.0384        |
| 1.04   | 6500  | 0.0346        |
| 1.12   | 7000  | 0.0374        |
| 1.2    | 7500  | 0.0375        |
| 1.28   | 8000  | 0.0405        |
| 1.3600 | 8500  | 0.0374        |
| 1.44   | 9000  | 0.0338        |
| 1.52   | 9500  | 0.0379        |
| 1.6    | 10000 | 0.0349        |
| 1.6800 | 10500 | 0.0358        |
| 1.76   | 11000 | 0.0391        |
| 1.8400 | 11500 | 0.0318        |
| 1.92   | 12000 | 0.0342        |
| 2.0    | 12500 | 0.0351        |
| 2.08   | 13000 | 0.0356        |
| 2.16   | 13500 | 0.0369        |
| 2.24   | 14000 | 0.0333        |
| 2.32   | 14500 | 0.0347        |
| 2.4    | 15000 | 0.0345        |
| 2.48   | 15500 | 0.0386        |
| 2.56   | 16000 | 0.0352        |
| 2.64   | 16500 | 0.0351        |
| 2.7200 | 17000 | 0.0336        |
| 2.8    | 17500 | 0.04          |
| 2.88   | 18000 | 0.0379        |
| 2.96   | 18500 | 0.0361        |


### Framework Versions
- Python: 3.9.5
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cpu
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->