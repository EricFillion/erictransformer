
| **Argument**          | **Description**                                                                                                                                   | **Default Value**     |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `lr`                  | learning rate                                                                                                                                     | `2e-5`                |
| `bs`                  | batch size                                                                                                                                        | `1`                   |
| `eval_bs`             | eval batch size                                                                                                                                   | `0`                   |
| `epochs`              | epochs                                                                                                                                            | `1`                   |
| `gas`                 | gradient accumulation steps                                                                                                                       | `1`                   |
| `optim`               | str = "adamw" # options adamw and sgd                                                                                                             | `"adamw"`             |
| `lr_sched`            | The learning rate scheduler. Either "constant" or "warmup_then_decay"                                                                             | `"constant"`          |
| `eval_steps`          | Number of steps until evaluation is performed.                                                                                                    | `256`                 |
| `log_steps`           | number of steps before logging occurs.                                                                                                            | `256`                 |
| `checkpoint_steps`    | Number of steps before checkpointing occurs. 0 results in no checkpointing.                                                                       | `0`                   |
| `save_best`           | If True, the model with the lowest eval loss is saved.                                                                                            | `False`               |
| `out_dir`             | A subdirectory will be created within the  output directory that shows the training progress, stores the tokenized data and contains checkpoints. | `"eric_transformer/"` |
| `run_name`            | Controls the name of the subdirectory that's created within the output directory.                                                                 | `""`                  |
| `seed`                | Controls randomness.                                                                                                                              | `42`                  |

