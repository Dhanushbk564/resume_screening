---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:16000
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: An API Developer creates and maintains application programming
    interfaces (APIs) that enable data exchange between different software applications
    or systems.
  sentences:
  - Electrical engineering Circuit design Electronics AutoCAD proficiency Problem-solving
  - API design and development RESTful API knowledge Security protocols (OAuth, JWT)
  - Social media management Content creation Audience engagement Marketing strategy
    Communication skills
- source_sentence: Family Law Attorneys deal with legal matters related to family
    relationships. They handle cases like divorce, child custody, adoption, and domestic
    disputes to provide legal guidance.
  sentences:
  - Technical troubleshooting Hardware and software support Customer service Problem-solving
    Communication
  - Retirement planning Social security Investment products Tax planning Estate planning
  - Family law Divorce proceedings Child custody Mediation Legal counseling Court
    representation
- source_sentence: The Agile Project Manager leads Agile teams, facilitating effective
    project development and delivery. They use Agile methodologies to manage tasks,
    prioritize work, and adapt to changing requirements while fostering collaboration
    among team members.
  sentences:
  - Agile methodologies Scrum or Kanban Team collaboration Agile tools (e.g., Jira)
  - Interaction design principles User behavior and psychology Wireframing and prototyping
    tools Animation and micro-interaction design Collaborative design processes
  - Automation and scripting (e.g., Python, Bash) Continuous Integration/Continuous
    Deployment (CI/CD) Containerization (e.g., Docker, Kubernetes) Infrastructure
    as Code (e.g., Terraform, Ansible) Cloud platforms (e.g., AWS, Azure, GCP) Monitoring
    and troubleshooting skills
- source_sentence: Technical Support Specialists assist customers or end-users with
    technical issues related to products or services. They provide troubleshooting,
    resolve problems, and offer technical guidance through various communication channels,
    ensuring customer satisfaction.
  sentences:
  - Interaction design principles User behavior and psychology Wireframing and prototyping
    tools Animation and micro-interaction design Collaborative design processes
  - Technical troubleshooting Customer support tools (e.g., Zendesk, Freshdesk) Communication
    skills Problem-solving and critical thinking Ticket management Knowledge base
    creation
  - Brand strategy Brand development Creative direction Brand management Market research
- source_sentence: Sustainable Design Specialists incorporate eco-friendly practices
    into architectural designs, promoting energy efficiency and environmental sustainability.
  sentences:
  - Records management Data entry and retrieval Attention to detail
  - Technical troubleshooting Customer service and communication Ticketing system
    usage Basic IT knowledge Problem-solving and critical-thinking skills
  - Sustainable design principles Energy efficiency LEED certification Green building
    materials Environmental impact assessment
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
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
    'Sustainable Design Specialists incorporate eco-friendly practices into architectural designs, promoting energy efficiency and environmental sustainability.',
    'Sustainable design principles Energy efficiency LEED certification Green building materials Environmental impact assessment',
    'Records management Data entry and retrieval Attention to detail',
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

* Size: 16,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            |
  | details | <ul><li>min: 14 tokens</li><li>mean: 31.75 tokens</li><li>max: 75 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 23.29 tokens</li><li>max: 79 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Wealth Advisors provide financial advice to clients. They assess financial goals, create investment strategies, and offer guidance on wealth management and financial planning.</code>                                                             | <code>Financial planning Investment knowledge Relationship management Communication skills Analytical skills</code>                                                                   |
  | <code>A Conference Manager coordinates and manages conferences, meetings, and events. They plan logistics, handle budgeting, liaise with vendors, and ensure the smooth execution of events, catering to the needs and expectations of attendees.</code> | <code>Event planning Conference logistics Budget management Vendor coordination Marketing and promotion Client relations</code>                                                       |
  | <code>User Interface Designers focus on the visual and interactive aspects of digital interfaces. They design layouts, buttons, and other elements to ensure a cohesive and visually appealing user interface.</code>                                    | <code>UI design principles and best practices Graphic design tools (e.g., Adobe Photoshop, Illustrator) Typography and color theory Visual design and layout Responsive design</code> |
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
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.5   | 500  | 0.0795        |
| 1.0   | 1000 | 0.0534        |
| 1.5   | 1500 | 0.0461        |
| 2.0   | 2000 | 0.0453        |
| 2.5   | 2500 | 0.041         |
| 3.0   | 3000 | 0.0398        |


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