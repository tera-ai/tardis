import random
import re
from functools import partial
from typing import List

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.experimental.pjit import pjit
from jax.interpreters import pxla
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder
from transformers import FlaxLogitsWarper


class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


class JaxDistributedConfig(object):
    """ Utility class for initializing JAX distributed. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.initialize_jax_distributed = False
        config.coordinator_address = placeholder(str)
        config.num_processes = placeholder(int)
        config.process_id = placeholder(int)
        config.local_device_ids = placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def initialize(cls, config):
        config = cls.get_default_config(config)
        if config.initialize_jax_distributed:
            if config.local_device_ids is not None:
                local_device_ids = [int(x) for x in config.local_device_ids.split(',')]
            else:
                local_device_ids = None

            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=config.process_id,
                local_device_ids=local_device_ids,
            )


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    """ JIT traceable version of FlaxLogitsWarper that performs temperature scaling."""
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, input_ids, scores, cur_len):
        return scores / jnp.clip(self.temperature, a_min=1e-8)


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """ Create pytree of sharding and gathering functions from pytree of
        partition specs.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor
        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )
        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()
        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )
        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))
        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def get_jax_mesh(axis_dims: str, names: List[str]) -> Mesh:
    if axis_dims.startswith("!"):
        # Allow splitting a physical mesh axis if needed
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ":" in axis_dims:
        dims = []
        dim_names = []

        for axis in axis_dims.split(","):
            name, dim = axis.split(":")
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)

        assert set(dim_names) == set(names)
    else:
        dims = [int(x) for x in axis_dims.split(",")]
        dim_names = names

    assert len(dims) == len(names)

    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape

    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)

    return Mesh(physical_mesh, dim_names)


def names_in_current_mesh(*names):
    """ Check if current mesh axes contain these names. """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_parition_spec(partition_specs) -> List[str]:
    """ Return axis names from partition specs. """
    names = set()

    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()

    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed: int) -> None:
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs) -> JaxRNG:
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, unreplicate=False, stack=False):
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss

class SpecialTokenGenerator:
    # Fixed reserved token markers
    BEGINNING_OF_SENTENCE = 0
    ENDING_OF_SENTENCE = 2

    # Hardcoded reserved token markers
    BEGINNING_OF_IMAGE = 8_197
    ENDING_OF_IMAGE = 8_196

    def __init__(
        self,
        min_latitude: float = 37.50555625328688,
        max_latitude: float = 37.57277786216565,
        latitude_precision: float = 0.00001,
        min_longitude: float = -122.3491638598312,
        max_longitude: float = -122.2491688227424,
        longitude_precision: float = 0.00001,
        min_month: int = 1,
        max_month: int = 12,
        month_precision: int = 1,
        min_year: int = 2000,
        max_year: int = 2030,
        year_precision: int = 1,
        min_move: float = 0.0,
        max_move: float = 50.0,
        move_precision: float = 0.1,
        min_heading: float = 0.0,
        max_heading: float = 360.0,
        heading_precision: float = 0.1,
        min_month_action: int = 0,
        max_month_action: int = 11,
        month_action_precision: int = 1,
        min_year_action: int = -30,
        max_year_action: int = 30,
        year_action_precision: int = 1,
    ) -> None:
        # 1. Latitude (after image)
        self.beginning_of_latitude_marker, self.ending_of_latitude_marker = self.compute_markers(
            self.BEGINNING_OF_IMAGE,
            min_latitude,
            max_latitude,
            latitude_precision,
        )

        # 2. Longitude
        self.beginning_of_longitude_marker, self.ending_of_longitude_marker = self.compute_markers(
            self.ending_of_latitude_marker,
            min_longitude,
            max_longitude,
            longitude_precision,
        )

        # 3. Month
        self.beginning_of_month_marker, self.ending_of_month_marker = self.compute_markers(
            self.ending_of_longitude_marker,
            min_month,
            max_month,
            month_precision,
        )

        # 4. Year
        self.beginning_of_year_marker, self.ending_of_year_marker = self.compute_markers(
            self.ending_of_month_marker,
            min_year,
            max_year,
            year_precision,
        )

        # 5. Move
        self.beginning_of_move_marker, self.ending_of_move_marker = self.compute_markers(
            self.ending_of_year_marker,
            min_move,
            max_move,
            move_precision,
        )

        # 6. Heading
        self.beginning_of_heading_marker, self.ending_of_heading_marker = self.compute_markers(
            self.ending_of_move_marker,
            min_heading,
            max_heading,
            heading_precision,
        )

        # 7. Month Action
        self.beginning_of_month_action_marker, self.ending_of_month_action_marker = self.compute_markers(
            self.ending_of_heading_marker,
            min_month_action,
            max_month_action,
            month_action_precision,
        )

        # 8. Year Action
        self.beginning_of_year_action_marker, self.ending_of_year_action_marker = self.compute_markers(
            self.ending_of_month_action_marker,
            min_year_action,
            max_year_action,
            year_action_precision,
        )

    def compute_markers(
        self,
        ending_of_prev_marker: int,
        min_value_of_marker: int,
        max_value_of_marker: int,
        precision_of_marker: int,
    ):
        beginning_of_marker = ending_of_prev_marker + 1
        range_span = max_value_of_marker - min_value_of_marker + precision_of_marker
        range_of_current_marker = int(range_span / precision_of_marker)
        ending_of_current_marker = beginning_of_marker + range_of_current_marker + 2
        return (
            beginning_of_marker,
            ending_of_current_marker,
        )

def get_modality_ranges():
    # Define special tokens
    special_tokens = SpecialTokenGenerator()
    special_tokens = {
        'bos': special_tokens.BEGINNING_OF_SENTENCE, 'eos': special_tokens.ENDING_OF_SENTENCE,
        'boi': special_tokens.BEGINNING_OF_IMAGE, 'eoi': special_tokens.ENDING_OF_IMAGE,
        'bla': special_tokens.beginning_of_latitude_marker, 'ela': special_tokens.ending_of_latitude_marker,
        'blo': special_tokens.beginning_of_longitude_marker, 'elo': special_tokens.ending_of_longitude_marker,
        'bmo': special_tokens.beginning_of_month_marker, 'emo': special_tokens.ending_of_month_marker,
        'bye': special_tokens.beginning_of_year_marker, 'eye': special_tokens.ending_of_year_marker,
        'bmv': special_tokens.beginning_of_move_marker, 'emv': special_tokens.ending_of_move_marker,
        'bhe': special_tokens.beginning_of_heading_marker, 'ehe': special_tokens.ending_of_heading_marker,
        'bma': special_tokens.beginning_of_month_action_marker, 'ema': special_tokens.ending_of_month_action_marker,
        'bya': special_tokens.beginning_of_year_action_marker, 'eya': special_tokens.ending_of_year_action_marker,
        'pad': 1
    }

    modality_ranges = {
      'text': {
          'range': (0, 2),  # bos, pad, eos
          'special': [special_tokens['bos'], special_tokens['pad'], special_tokens['eos']]
      },
      'image': {
          'range': (3, special_tokens['eoi']), # chameleon is annoying
          'special': [special_tokens['boi'], special_tokens['eoi']]
      },
      'latitude': {
          'range': (special_tokens['bla'], special_tokens['ela']),
          'special': [special_tokens['bla'], special_tokens['ela']]
      },
      'longitude': {
          'range': (special_tokens['blo'], special_tokens['elo']),
          'special': [special_tokens['blo'], special_tokens['elo']]
      },
      'month': {
          'range': (special_tokens['bmo'], special_tokens['emo']),
          'special': [special_tokens['bmo'], special_tokens['emo']]
      },
      'year': {
          'range': (special_tokens['bye'], special_tokens['eye']),
          'special': [special_tokens['bye'], special_tokens['eye']]
      },
      'move': {
          'range': (special_tokens['bmv'], special_tokens['emv']),
          'special': [special_tokens['bmv'], special_tokens['emv']]
      },
      'heading': {
          'range': (special_tokens['bhe'], special_tokens['ehe']),
          'special': [special_tokens['bhe'], special_tokens['ehe']]
      },
      'month_action': {
          'range': (special_tokens['bma'], special_tokens['ema']),
          'special': [special_tokens['bma'], special_tokens['ema']]
      },
      'year_action': {
          'range': (special_tokens['bya'], special_tokens['eya']),
          'special': [special_tokens['bya'], special_tokens['eya']]
      }
    }

    return modality_ranges

def get_auto_weight(modality_ranges, ceiling=False, vocab=29163):
    weights = {}
    for modality, values in modality_ranges.items():
        min_value, max_value = values['range']
        scale = max_value - min_value
        norm_scale = (scale / vocab)*10
        norm_scale += 1
        if ceiling:
          norm_scale = np.ceil(norm_scale)
        weights[modality] = norm_scale
    return weights

def cross_entropy_loss_multimodal(logits, tokens, valid=None,
                                   exclude_special=False, modality_weight=None):
    """
    Multi-modal cross entropy loss and accuracy calculation.
    
    Args:
        logits: Model predictions
        tokens: Ground truth tokens
        special_tokens: Dictionary containing token markers for each modality
        valid: Optional mask for valid tokens
        modality_weight: Optional dictionary of scaling factors for each modality's loss
    """
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    
    # Define modality ranges
    modality_ranges = get_modality_ranges()

    # Default scale is 1.0 for all modalities if not specified
    if modality_weight is None:
        modality_weight = {modality: 1.0 for modality in modality_ranges.keys()}
    elif modality_weight == 'auto':
        modality_weight = get_auto_weight(modality_ranges, ceiling=False)

    # Create masks for each modality
    modality_masks = {}
    for modality, values in modality_ranges.items():
        start, end = values['range']
        if exclude_special:
            if modality == 'image':
                # chameleon makes this weird -- end ie 8196
                modality_masks[modality] = (tokens >= 3) & (tokens < 8196) # end
            else:
                modality_masks[modality] = (tokens > start) & (tokens < end)
        else:
            if modality == 'image':
                # chameleon makes this weird -- start ie 8197
                modality_masks[modality] = (tokens >= 3) & (tokens <= 8197) # start
            else:
                modality_masks[modality] = (tokens >= start) & (tokens <= end)

    # Calculate valid lengths for each modality
    valid_lengths = {}
    for modality, mask in modality_masks.items():
        valid_lengths[modality] = jnp.maximum(
            jnp.sum(valid * mask, axis=-1),
            1e-10  # Prevent division by zero
        )

    # Compute log probabilities
    logits = logits.astype(jnp.float32)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    
    # Mask invalid tokens
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    # Calculate losses and accuracies for each modality
    losses = {}
    accuracies = {}
    correct = jnp.where(valid > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False))
    
    for modality, mask in modality_masks.items():
        # Compute modality-specific log probabilities
        modality_log_prob = jnp.where(mask & (valid > 0.0), token_log_prob, jnp.array(0.0))
        
        # Compute loss
        modality_loss = -jnp.mean(
            jnp.sum(modality_log_prob, axis=-1) / valid_lengths[modality]
        ) * modality_weight[modality]
        
        losses[modality] = modality_loss
        
        # Compute accuracy
        correct_modality = jnp.where(mask & (valid > 0.0), correct, jnp.array(False))
        accuracy = jnp.mean(
            jnp.sum(correct_modality, axis=-1) / valid_lengths[modality]
        )
        
        accuracies[modality] = accuracy

    # Compute total loss as sum of all modality losses
    if modality_weight == 'auto':
        total_loss = sum(losses.values()) / 20
    else:
        total_loss = sum(losses.values()) / len(losses)

    # MARK: New 5x scaling
    total_loss = total_loss * 5

    # Return metrics
    metrics = {
        **{f'{modality}_loss': loss for modality, loss in losses.items()},
        **{f'{modality}_accuracy': acc for modality, acc in accuracies.items()}
    }

    return total_loss, metrics

def cross_entropy_loss_and_accuracy_with_image(logits, tokens, valid=None, lang_scale=3):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)

    # Separate valid text lengths for language and image
    # raised smallest token to 3 because chameleon considers 0 <bos>, 1 <pad> & 2 <eos>
    is_image_token = (tokens >= 3) & (tokens <= 8197) # 8197 is <boi>, 8196 is <eoi>
    is_language_token = ~is_image_token

    valid_language_length = jnp.maximum(jnp.sum(valid * is_language_token, axis=-1), 1e-10)
    valid_image_length = jnp.maximum(jnp.sum(valid * is_image_token, axis=-1), 1e-10)

    # Logits conversion for numerical stability
    logits = logits.astype(jnp.float32)

    # Compute log probabilities for tokens
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )

    # Mask token log probabilities based on valid tokens
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    # Separate language and image losses
    language_log_prob = jnp.where(is_language_token & (valid > 0.0), token_log_prob, jnp.array(0.0))
    image_log_prob = jnp.where(is_image_token & (valid > 0.0), token_log_prob, jnp.array(0.0))

    # Compute losses
    language_loss = -jnp.mean(jnp.sum(language_log_prob, axis=-1) / valid_language_length) * lang_scale
    image_loss = -jnp.mean(jnp.sum(image_log_prob, axis=-1) / valid_image_length)

    # Overall loss as a sum (or other desired combination)
    total_loss = language_loss + image_loss

    # Compute accuracy
    correct = jnp.where(valid > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False))
    correct_language = jnp.where(is_language_token & (valid > 0.0), correct, jnp.array(False))
    correct_image = jnp.where(is_image_token & (valid > 0.0), correct, jnp.array(False))

    # Separate accuracies for language and image tokens
    language_accuracy = jnp.mean(jnp.sum(correct_language, axis=-1) / valid_language_length)
    image_accuracy = jnp.mean(jnp.sum(correct_image, axis=-1) / valid_image_length)

    return total_loss, {
        'language_loss': language_loss,
        'image_loss': image_loss,
        'language_accuracy': language_accuracy,
        'image_accuracy': image_accuracy
    }



def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def float_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(
        partial(float_tensor_to_dtype, dtype=dtype), tree
    )


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def flatten_tree(xs, is_leaf=None, sep=None):
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    return named_tree_map(get_partition_spec, params, sep='/')


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return named_tree_map(decay, params, sep='/')

    return weight_decay_mask


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)
