# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from examples.simultaneous_translation.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
    TransformerMonotonicEncoderLayer,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    tiny_architecture
)
from torch import Tensor
import logging
from typing import Optional

import torch  # noqa 42
from torch import nn


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
READ_ACTION = 0
WRITE_ACTION = 1

TransformerMonotonicDecoderOut = NamedTuple(
    "TransformerMonotonicDecoderOut",
    [
        ("action", int),
        ("p_choose", Optional[Tensor]),
        ("attn_list", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("encoder_out", Optional[Dict[str, List[Tensor]]]),
        ("encoder_padding_mask", Optional[Tensor]),
    ],
)


@register_model("transformer_unidirectional")
class TransformerUnidirectionalModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)


@register_model("transformer_monotonic")
class TransformerModelSimulTrans(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

from fairseq.models.transformer.transformer_config import (
    TransformerConfig, EncDecBaseConfig, DecoderConfig, QuantNoiseConfig
    )



class TransformerMonotonicDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        if dictionary is None:
            dictionary = [i for i in range(103)]
            self.dictionary = dictionary
        if embed_tokens is None:
            embed_tokens = nn.Embedding(103, 128, padding_idx=1)
        if args is None:
            args = TransformerConfig(
                _name='transformer_monotonic_iwslt_de_en',
                activation_fn='relu',
                dropout=0.3,
                attention_dropout=0.0,
                activation_dropout=0.0,
                adaptive_input=False,
                encoder=EncDecBaseConfig(
                    _name='transformer_monotonic_iwslt_de_en',
                    embed_path=None,
                    embed_dim=512,
                    ffn_embed_dim=1024,
                    layers=6,
                    attention_heads=4,
                    normalize_before=False,
                    learned_pos=False,
                    layerdrop=0,
                    layers_to_keep=None
                ),
                max_source_positions=1024,
                decoder=DecoderConfig(
                    _name='transformer_monotonic_iwslt_de_en',
                    embed_path=None,
                    embed_dim=512,
                    ffn_embed_dim=1024,
                    layers=6,
                    attention_heads=4,
                    normalize_before=False,
                    learned_pos=False,
                    layerdrop=0,
                    layers_to_keep=None,
                    input_dim=512,
                    output_dim=512
                ),
                max_target_positions=1024,
                share_decoder_input_output_embed=False,
                share_all_embeddings=False,
                no_token_positional_embeddings=False,
                adaptive_softmax_cutoff=None,
                adaptive_softmax_dropout=0,
                adaptive_softmax_factor=4,
                layernorm_embedding=False,
                tie_adaptive_weights=False,
                tie_adaptive_proj=False,
                no_scale_embedding=False,
                checkpoint_activations=False,
                offload_activations=False,
                no_cross_attention=False,
                cross_self_attention=False,
                quant_noise=QuantNoiseConfig(
                    _name='transformer_monotonic_iwslt_de_en',
                    pq=0,
                    pq_block_size=8,
                    scalar=0
                ),
                min_params_to_wrap=100000000,
                char_inputs=False,
                relu_dropout=0.0,
                base_layers=0,
                base_sublayers=1,
                base_shuffle=1,
                export=False,
                no_decoder_final_norm=False
            )
        args.simul_type="infinite_lookback"
        args.noise_type="flat"
        args.noise_mean=0.0
        args.noise_var=1.0
        args.energy_bias_init=-2.0
        args.energy_bias=False
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicDecoderLayer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.policy_criterion = getattr(args, "policy_criterion", "any")
        self.num_updates = None

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict
        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
            print("encoder_padding_mask", encoder_padding_mask)
            print("encoder_padding_mask", encoder_padding_mask.size())
            print("encoder padding mask all False", not encoder_padding_mask)
        else:
            encoder_padding_mask = None

        # return x, encoder_out, encoder_padding_mask
        return x, encoder_out, None

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clean_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clean cache in the monotonic layers.
        The cache is generated because of a forward pass of decoder has run but no prediction,
        so that the self attention key value in decoder is written in the incremental state.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def make_masks_for_mt(self, src, tgt, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : torch.Tensor
            The sequence to the encoder (required).
        tgt : torch.Tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).

        Returns
        -------
        src_key_padding_mask : torch.Tensor
            Timesteps to mask due to padding
        tgt_key_padding_mask : torch.Tensor
            Timesteps to mask due to padding
        src_mask : torch.Tensor
            Timesteps to mask for causality
        tgt_mask : torch.Tensor
            Timesteps to mask for causality
        """
        src_key_padding_mask = None
        if self.training:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)

        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        if (torch.isnan(prev_output_tokens).any()):
            raise KeyboardInterrupt("extract_features prev")
        
        if (torch.isnan(encoder_out).any()):
            raise KeyboardInterrupt("extract_features encoder")
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []

        p_choose = torch.tensor([1.0])

        if (torch.isnan(x).any()):
            raise KeyboardInterrupt("extract_features Decoder")
        for i, layer in enumerate(self.layers):
            # raise KeyboardInterrupt()
            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
            )

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None:
                if_online = incremental_state["online"]["only"]
                assert if_online is not None
                if if_online.to(torch.bool):
                    # Online indicates that the encoder states are still changing
                    assert attn is not None
                    if self.policy_criterion == "any":
                        # Any head decide to read than read
                        head_read = layer.encoder_attn._get_monotonic_buffer(incremental_state)["head_read"]
                        assert head_read is not None
                        if head_read.any():
                            # We need to prune the last self_attn saved_state
                            # if model decide not to read
                            # otherwise there will be duplicated saved_state
                            self.clean_cache(incremental_state, i + 1)

                            return x, TransformerMonotonicDecoderOut(
                                action=0,
                                p_choose=p_choose,
                                attn_list=None,
                                encoder_out=None,
                                encoder_padding_mask=None,
                            )

        x = self.post_attention(x)

        return x, TransformerMonotonicDecoderOut(
            action=1,
            p_choose=p_choose,
            attn_list=attn_list,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
        )


@register_model_architecture("transformer_monotonic", "transformer_monotonic")
def base_monotonic_architecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, "encoder_unidirectional", False)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_iwslt_de_en"
)
def transformer_monotonic_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    base_monotonic_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_de_big"
)
def transformer_monotonic_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_fr_big"
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    transformer_monotonic_vaswani_wmt_en_fr_big(args)


@register_model_architecture(
    "transformer_unidirectional", "transformer_unidirectional_iwslt_de_en"
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("transformer_monotonic", "transformer_monotonic_tiny")
def monotonic_tiny_architecture(args):
    tiny_architecture(args)
    base_monotonic_architecture(args)
