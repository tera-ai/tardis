import os
import pprint
import socket
import time

import jax
import jax.numpy as jnp
import mlxu
from flax.training.train_state import TrainState
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tqdm import tqdm
from tqdm import trange

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import JaxDistributedConfig
from EasyLM.jax_utils import JaxRNG
from EasyLM.jax_utils import average_metrics
from EasyLM.jax_utils import cross_entropy_loss_multimodal
from EasyLM.jax_utils import get_float_dtype_by_name
from EasyLM.jax_utils import get_weight_decay_mask
from EasyLM.jax_utils import global_norm
from EasyLM.jax_utils import make_shard_and_gather_fns
from EasyLM.jax_utils import match_partition_rules
from EasyLM.jax_utils import next_rng
from EasyLM.jax_utils import set_random_seed
from EasyLM.jax_utils import with_sharding_constraint
from EasyLM.models.llama.llama_model import FlaxLLaMAForCausalLMModule
from EasyLM.models.llama.llama_model import LLaMAConfig


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim="1,-1,1",
    dtype="fp32",
    total_steps=10000,
    load_llama_config="",
    update_llama_config="",
    load_checkpoint="",
    load_dataset_state="",
    load_metadata="",
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    total_eval_steps=-1,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    # NOTE: Kubernetes Job-specific flags
    kubernetes=False,
    job_name="",
    sub_domain="",
    coordinator_port=1234,
)


def _get_coordinator_ip_address(job_name: str, sub_domain: str, retry_count: int = 120) -> str:
    assert job_name != "", "job_name may not be left in blank"
    assert sub_domain != "", "sub_domain may not be left in blank"

    coordinator_fqdn = f"{FLAGS.job_name}-0.{FLAGS.sub_domain}"
    print(f"Coordinator host name: {coordinator_fqdn}") 

    for _ in range(retry_count):
        try:
            time.sleep(1)
            coordinator_ip_address = socket.gethostbyname(coordinator_fqdn)
        except socket.gaierror:
            print(f"Failed to resolve: {coordinator_fqdn}. Trying again in a second ...") 
        else:
            break

    print(f"Coordinator IP address: {coordinator_ip_address}")

    return coordinator_ip_address


def main(argv) -> None:
    # If running in a Kubernetes Job context,
    # dynamically set flags depending on the environment it's running in
    if FLAGS.kubernetes:
        coordinator_ip_address = _get_coordinator_ip_address(FLAGS.job_name, FLAGS.sub_domain)
        coordinator_address = f"{coordinator_ip_address}:{FLAGS.coordinator_port}"

        FLAGS.jax_distributed.coordinator_address = coordinator_address
        FLAGS.jax_distributed.process_id = int(os.environ["JOB_COMPLETION_INDEX"])
        assert FLAGS.jax_distributed.num_processes >= 1, "num_processes should be greater or equal to 1"

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    print("JAX Devices:", jax.devices())
    print("JAX DeviceCount:", jax.device_count())

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(
        FLAGS.train_dataset, tokenizer, device_count=jax.device_count()
    )

    if FLAGS.load_dataset_state != "":
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != "":
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != "":
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    # llama_config.update(dict(
    #     bos_token_id=dataset.tokenizer.bos_token_id,
    #     eos_token_id=dataset.tokenizer.eos_token_id,
    # ))
    # if llama_config.vocab_size < dataset.vocab_size:
    #     llama_config.update(dict(vocab_size=dataset.vocab_size))

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))

        def loss_and_accuracy(params):
            logits = model.apply(
                params, batch["input_tokens"], deterministic=False,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_multimodal(
                logits, batch["target_tokens"], batch["loss_masks"]
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, train_metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            language_loss=train_metrics["text_loss"],
            image_loss=train_metrics["image_loss"],
            lat_loss=train_metrics["latitude_loss"],
            lon_loss=train_metrics["longitude_loss"],
            month_loss=train_metrics["month_loss"],
            year_loss=train_metrics["year_loss"],
            move_loss=train_metrics["move_loss"],
            heading_loss=train_metrics["heading_loss"],
            month_action_loss=train_metrics["month_action_loss"],
            year_action_loss=train_metrics["year_action_loss"],

            language_accuracy=train_metrics["text_accuracy"],
            image_accuracy= train_metrics["image_accuracy"],
            lat_accuracy=train_metrics["latitude_accuracy"],
            lon_accuracy=train_metrics["longitude_accuracy"],
            month_accuracy=train_metrics["month_accuracy"],
            year_accuracy=train_metrics["year_accuracy"],
            move_accuracy=train_metrics["move_accuracy"],
            head_accuracy=train_metrics["heading_accuracy"],
            month_action_accuracy=train_metrics["month_action_accuracy"],
            year_action_accuracy=train_metrics["year_action_accuracy"],

            learning_rate=optimizer_info["learning_rate_schedule"](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            train_state.params, batch["input_tokens"], deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        val_loss, val_metrics = cross_entropy_loss_multimodal(
            logits, batch["target_tokens"], batch["loss_masks"]
        )
        metrics = dict(
            eval_loss=val_loss,
            eval_language_loss=val_metrics["text_loss"],
            eval_image_loss=val_metrics["image_loss"],
            eval_lat_loss=val_metrics["latitude_loss"],
            eval_lon_loss=val_metrics["longitude_loss"],
            eval_month_loss=val_metrics["month_loss"],
            eval_year_loss=val_metrics["year_loss"],
            eval_move_loss=val_metrics["move_loss"],
            eval_heading_loss=val_metrics["heading_loss"],
            eval_month_action_loss=val_metrics["month_action_loss"],
            eval_year_action_loss=val_metrics["year_action_loss"],

            eval_anguage_accuracy=val_metrics["text_accuracy"],
            eval_image_accuracy= val_metrics["image_accuracy"],
            eval_lat_accuracy=val_metrics["latitude_accuracy"],
            eval_lon_accuracy=val_metrics["longitude_accuracy"],
            eval_month_accuracy=val_metrics["month_accuracy"],
            eval_year_accuracy=val_metrics["year_accuracy"],
            eval_move_accuracy=val_metrics["move_accuracy"],
            eval_head_accuracy=val_metrics["heading_accuracy"],
            eval_month_action_accuracy=val_metrics["month_action_accuracy"],
            eval_year_action_accuracy=val_metrics["year_action_accuracy"],
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)

    with mesh:
        train_state, restored_params = None, None

        if FLAGS.load_checkpoint != "":
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

            # Get correct step from the metadata, if provided.
            # Otherwise, the `train_state.step` is initialized at 0.
            if FLAGS.load_metadata != "":
                loaded_step = mlxu.load_pickle(FLAGS.load_metadata)["step"] + 1
                train_state = train_state.replace(step=loaded_step)
                print(f"Resuming at step {loaded_step}")

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps != 0 and step % FLAGS.eval_steps == 0:
                    eval_metric_list = []
                    for _ in tqdm(range(FLAGS.total_eval_steps)):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)

                # NOTE: Only update tqdm progress on master JAX process
                if jax.process_index() == 0:
                    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
