"""Rejection sampling algorithms for constrained generation."""

import time
from typing import List, Optional
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    InfNanRemoveLogitsProcessor,
)

from celeste.samplers.base import BaseSampler, SamplingResult
from celeste.utils.oracle_logits_processor import OracleLogitsProcessor
from celeste.utils.scoring import get_seq_logprob_from_scores


class RS(BaseSampler):
    """Rejection Sampling (RS).
    
    Basic rejection sampling without learning from rejected samples.
    """
    
    def __init__(self, llm, grammar, max_new_tokens: int = 512):
        """Initialize RS sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            max_new_tokens: Maximum tokens to generate.
        """
        super().__init__(llm, grammar, max_new_tokens)
        self.learn_level = 0
        self.constrain_first = False
    
    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        n_steps: int = 2000,
    ) -> List[SamplingResult]:
        """Generate samples using rejection sampling.
        
        Args:
            prompt: Input prompt.
            n_samples: Number of successful samples to generate.
            n_steps: Maximum generation attempts.
            
        Returns:
            List of successful sampling results.
        """
        prompt_ids = self._encode_prompt(prompt)
        results = []
        
        # Initialize logits processor
        logits_processor = OracleLogitsProcessor(
            tokenizer=self.llm.tokenizer,
            grammar_constraint=self.grammar.recognizer,
            device=self.llm.device,
            learn_level=self.learn_level,
            constrain_first=self.constrain_first,
        )
        
        for step in range(n_steps):
            if len(results) >= n_samples:
                break
            
            try:
                result = self._generate_one(prompt_ids, logits_processor)
                results.append(result)
            except ValueError:
                # Rejected sample, continue
                continue
        
        return results
    
    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        logits_processor: OracleLogitsProcessor,
    ) -> SamplingResult:
        """Generate a single sample.
        
        Args:
            prompt_ids: Encoded prompt.
            logits_processor: Logits processor for constraints.
            
        Returns:
            Sampling result.
            
        Raises:
            ValueError: If sample violates constraints.
        """
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=None,
        )
        
        logits_processor.reset()
        logits_processor_list = LogitsProcessorList([
            logits_processor,
            InfNanRemoveLogitsProcessor(),
        ])
        
        output = self.llm.model.generate(
            prompt_ids,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )
        
        output_ids = output.sequences
        raw_logprob = logits_processor.generation_ended(output_ids)
        
        # Extract generated tokens (excluding prompt)
        generated_ids = output_ids[:, prompt_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)
        
        # Calculate constrained log probability
        constrained_logprob = get_seq_logprob_from_scores(
            output_scores,
            generated_ids,
            self.llm.tokenizer.eos_token_id,
        ).item()
        
        # Prepare result
        token_ids = generated_ids[0].tolist()
        tokens = [self.llm.tokenizer.decode([tid]) for tid in token_ids]
        text = self.llm.tokenizer.decode(generated_ids[0])
        
        return SamplingResult(
            tokens=tokens,
            token_ids=token_ids,
            text=text,
            raw_logprob=raw_logprob,
            constrained_logprob=constrained_logprob,
            success=True,
        )


class ARS(RS):
    """Adaptive Rejection Sampling (ARS).
    
    Learns from rejected samples to improve efficiency.
    """
    
    def __init__(self, llm, grammar, max_new_tokens: int = 512):
        """Initialize ARS sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            max_new_tokens: Maximum tokens to generate.
        """
        super().__init__(llm, grammar, max_new_tokens)
        self.learn_level = 2


class RSFT(RS):
    """Rejection Sampling with constrained First Token (RSFT).
    
    Constrains the first token to valid grammar tokens.
    """
    
    def __init__(self, llm, grammar, max_new_tokens: int = 512):
        """Initialize RSFT sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            max_new_tokens: Maximum tokens to generate.
        """
        super().__init__(llm, grammar, max_new_tokens)
        self.learn_level = 0
        self.constrain_first = True


class CARS(RS):
    """Constrained Adaptive Rejection Sampling (CARS).
    
    Combines adaptive learning with first token constraints for optimal efficiency.
    """
    
    def __init__(self, llm, grammar, max_new_tokens: int = 512):
        """Initialize CARS sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            max_new_tokens: Maximum tokens to generate.
        """
        super().__init__(llm, grammar, max_new_tokens)
        self.learn_level = 3
        self.constrain_first = True