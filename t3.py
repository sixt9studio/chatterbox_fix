# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import (
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinPLogitsWarper,
)

from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..utils import AttrDict


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        # ✅ SAFE CONFIG OVERRIDE (NO DUPLICATES)
        cfg_dict = dict(LLAMA_CONFIGS[hp.llama_config_name])
        cfg_dict["attn_implementation"] = "eager"

        self.cfg = LlamaConfig(**cfg_dict)
        self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        if hp.input_pos_emb == "learned":
            self.text_pos_emb = LearnedPositionEmbeddings(
                hp.max_text_tokens + 2, self.dim
            )
            self.speech_pos_emb = LearnedPositionEmbeddings(
                hp.max_speech_tokens + 4, self.dim
            )

        self.text_head = nn.Linear(self.dim, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.dim, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = (
                self.speech_emb(t3_cond.cond_prompt_speech_tokens)
                + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            )
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens)
        if cfg_weight > 0.0:
            text_emb[1].zero_()

        speech_emb = self.speech_emb(speech_tokens)
        if self.hp.input_pos_emb == "learned":
            text_emb += self.text_pos_emb(text_tokens)
            speech_emb += self.speech_pos_emb(speech_tokens)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.stack(
            [torch.cat((ce, te, se)) for ce, te, se in zip(cond_emb, text_emb, speech_emb)]
        )
        return embeds, cond_emb.size(1)

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,
        max_new_tokens: int,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)

        if initial_speech_tokens is None:
            initial_speech_tokens = (
                self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            )

        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        if not self.compiled:
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9,
                    eos_idx=self.hp.stop_speech_token,
                )

            self.patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.compiled = True

        bos = torch.tensor([[self.hp.start_speech_token]], device=self.device)
        bos_embed = self.speech_emb(bos) + self.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past = output.past_key_values
        generated = []

        top_p_warper = TopPLogitsWarper(top_p)
        min_p_warper = MinPLogitsWarper(min_p)
        rep_penalty = RepetitionPenaltyLogitsProcessor(repetition_penalty)

        for i in tqdm(range(max_new_tokens), desc="Sampling"):
            logits = output.logits[:, -1, :]
            cond, uncond = logits[:1], logits[1:]
            logits = cond + cfg_weight * (cond - uncond)

            logits = rep_penalty(bos, logits)
            logits /= temperature
            logits = min_p_warper(bos, logits)
            logits = top_p_warper(bos, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            if next_token.item() == self.hp.stop_speech_token:
                break

            next_embed = self.speech_emb(next_token)
            next_embed += self.speech_pos_emb.get_fixed_embedding(i + 1)
            next_embed = torch.cat([next_embed, next_embed])

            output = self.patched_model(
                inputs_embeds=next_embed,
                past_key_values=past,
                output_attentions=True,
                return_dict=True,
            )
            past = output.past_key_values

        return torch.cat(generated, dim=1)
