import time
from typing import List, Optional, Literal
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    LogitsProcessor,
    InfNanRemoveLogitsProcessor,
)

from celeste.samplers.base import BaseSampler, SamplingResult
from celeste.utils.grammar_logits_processor import GrammarLogitsProcessor
from celeste.utils.scoring import get_seq_logprob_from_scores


class _RestrictorLogitsProcessor(LogitsProcessor):
    """Logits processor that restricts generation to follow a specific sequence.
    
    Used internally for computing exact log probabilities of sequences.
    """
    
    def __init__(self, prompt_len: int, answer_ids: torch.LongTensor):
        """Initialize restrictor.
        
        Args:
            prompt_len: Length of the prompt.
            answer_ids: Token IDs to restrict generation to.
        """
        self.prompt_len = prompt_len
        self.answer_ids = answer_ids
        self.result = torch.empty(len(answer_ids))
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Restrict logits to force specific token."""
        pos = input_ids.size(1) - self.prompt_len
        assert 0 <= pos < self.answer_ids.size(0)
        
        if pos > 0:
            assert input_ids[0, -1] == self.answer_ids[pos - 1]
        
        # Store log probability of the forced token
        logprobs = torch.log_softmax(scores.to(torch.get_default_dtype()), dim=-1)
        self.result[pos] = logprobs[0][self.answer_ids[pos]]
        
        # Force only the target token to be possible
        scores = scores.clone()
        scores.fill_(float('-inf'))
        scores[0, self.answer_ids[pos]] = 0
        
        return scores


class MCMC(BaseSampler):
    """MCMC sampling with different proposal distributions.
    
    Implements three MCMC variants from the paper "Constrained Sampling for 
    Language Models Should Be Easy: An MCMC Perspective":
    - prefix: Uniform resampling over all positions
    - priority: Entropy-weighted resampling
    - restart: Always resample from the beginning
    
    Args:
        llm: LLM instance.
        grammar: Grammar instance.
        variant: MCMC variant ("prefix", "priority", or "restart").
        max_new_tokens: Maximum tokens to generate.
    """
    
    def __init__(
        self,
        llm,
        grammar,
        variant: Literal["prefix", "priority", "restart"] = "prefix",
        max_new_tokens: int = 512,
    ):
        """Initialize MCMC sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            variant: MCMC proposal strategy.
            max_new_tokens: Maximum tokens to generate.
        """
        super().__init__(llm, grammar, max_new_tokens)
        
        if variant not in ["prefix", "priority", "restart"]:
            raise ValueError(
                f"Invalid variant '{variant}'. Must be 'prefix', 'priority', or 'restart'."
            )
        
        self.variant = variant
    
    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        n_steps: int = 10,
    ) -> List[SamplingResult]:
        """Generate samples using MCMC.
        
        Args:
            prompt: Input prompt.
            n_samples: Number of independent chains to run.
            n_steps: Number of MCMC steps per chain.
            
        Returns:
            List of final sampling results (one per chain).
        """
        prompt_ids = self._encode_prompt(prompt)
        results = []
        
        for sample_idx in range(n_samples):
            # Initialize chain with unconstrained sample
            current_ids, current_scores = self._generate_constrained(
                prompt_ids=prompt_ids,
                prefix_ids=None,
            )
            
            # Run MCMC chain
            for step in range(n_steps):
                # Propose new sequence
                next_ids, next_scores, forward_logprob = self._propose_next_sequence(
                    prompt_ids=prompt_ids,
                    current_ids=current_ids,
                    current_scores=current_scores,
                )
                
                # Compute reverse proposal probability
                reverse_logprob = self._compute_proposal_logprob(
                    current_ids=next_ids,
                    current_scores=next_scores,
                    next_ids=current_ids,
                    next_scores=current_scores,
                )
                
                # Compute target log probabilities
                current_target_logprob = self._compute_sequence_logprob(
                    prompt_ids, current_ids
                )
                next_target_logprob = self._compute_sequence_logprob(
                    prompt_ids, next_ids
                )
                
                # Metropolis-Hastings acceptance
                log_accept_ratio = (
                    next_target_logprob - current_target_logprob +
                    reverse_logprob - forward_logprob
                )
                
                if np.log(np.random.random()) < log_accept_ratio:
                    # Accept proposal
                    current_ids = next_ids
                    current_scores = next_scores
            
            # Create final result
            token_ids = current_ids[0].tolist()
            tokens = [self.llm.tokenizer.decode([tid]) for tid in token_ids]
            text = self.llm.tokenizer.decode(current_ids[0])
            
            logprob = self._compute_sequence_logprob(prompt_ids, current_ids)
            
            results.append(SamplingResult(
                tokens=tokens,
                token_ids=token_ids,
                text=text,
                raw_logprob=logprob,
                constrained_logprob=logprob,
                success=True,
            ))
        
        return results
    
    def _generate_constrained(
        self,
        prompt_ids: torch.Tensor,
        prefix_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a sequence with grammar constraints.
        
        Args:
            prompt_ids: Encoded prompt.
            prefix_ids: Optional prefix to condition on.
            
        Returns:
            Tuple of (generated_ids, scores).
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
        
        # Setup grammar constraint
        self.grammar.reset()
        grammar_processor = GrammarLogitsProcessor(
            tokenizer=self.llm.tokenizer,
            grammar_constraint=self.grammar.recognizer,
            device=self.llm.device,
            prompt_length=len(prompt_ids[0]),
        )
        
        logits_processor_list = LogitsProcessorList([
            grammar_processor,
            InfNanRemoveLogitsProcessor(),
        ])
        
        # Concatenate prompt and prefix if provided
        input_ids = prompt_ids
        if prefix_ids is not None:
            input_ids = torch.cat([prompt_ids, prefix_ids], dim=-1)
        
        # Generate
        output = self.llm.model.generate(
            input_ids,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )
        
        # Extract generated tokens (excluding input)
        output_ids = output.sequences[:, input_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)
        
        return output_ids, output_scores
    
    def _compute_resampling_distribution(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distribution over resampling positions.
        
        Args:
            current_ids: Current sequence token IDs.
            current_scores: Current sequence scores.
            
        Returns:
            Probability distribution over positions.
        """
        seq_len = current_ids.shape[1]
        
        if self.variant == "restart":
            # Always resample from beginning
            distr = torch.zeros(seq_len, dtype=torch.float32)
            distr[0] = 1.0
        
        elif self.variant == "prefix":
            # Uniform distribution over all positions
            distr = torch.ones(seq_len) / seq_len
        
        elif self.variant == "priority":
            # Entropy-weighted distribution
            logprobs = torch.log_softmax(current_scores, dim=-1)
            
            # Compute entropy at each position
            mask = torch.isfinite(logprobs)
            probs = torch.exp(logprobs)
            masked_contrib = torch.where(
                mask,
                probs * logprobs,
                torch.zeros_like(probs),
            )
            entropies = -torch.sum(masked_contrib, dim=-1)
            
            # Convert to probability distribution
            # Subtract 1 to zero out entropies of 0
            distr = torch.exp(entropies[0]) - 1
            distr = distr / torch.sum(distr)
        
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        distr = distr.unsqueeze(0)
        assert distr.shape == current_ids.shape
        assert torch.allclose(distr.sum(), torch.tensor(1.0))
        
        return distr
    
    def _propose_next_sequence(
        self,
        prompt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Propose a new sequence via MCMC proposal.
        
        Args:
            prompt_ids: Encoded prompt.
            current_ids: Current sequence token IDs.
            current_scores: Current sequence scores.
            
        Returns:
            Tuple of (next_ids, next_scores, proposal_logprob).
        """
        # Sample resampling position
        resample_distr = self._compute_resampling_distribution(
            current_ids, current_scores
        )
        resample_idx = np.random.choice(
            len(current_ids[0]),
            p=resample_distr[0].cpu().numpy(),
        )
        
        # Extract prefix
        prefix_ids = current_ids[:, :resample_idx]
        prefix_scores = current_scores[:, :resample_idx]
        
        # Generate suffix
        resample_ids, resample_scores = self._generate_constrained(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
        )
        
        # Concatenate
        next_ids = torch.cat([prefix_ids, resample_ids], dim=-1)
        next_scores = torch.cat([prefix_scores, resample_scores], dim=1)
        
        # Compute proposal probability
        proposal_logprob = self._compute_proposal_logprob(
            current_ids=current_ids,
            current_scores=current_scores,
            next_ids=next_ids,
            next_scores=next_scores,
        )
        
        return next_ids, next_scores, proposal_logprob
    
    def _compute_proposal_logprob(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
        next_ids: torch.Tensor,
        next_scores: torch.Tensor,
    ) -> float:
        """Compute log probability of proposing next_ids from current_ids.
        
        Args:
            current_ids: Current sequence token IDs.
            current_scores: Current sequence scores.
            next_ids: Proposed sequence token IDs.
            next_scores: Proposed sequence scores.
            
        Returns:
            Log probability of the proposal.
        """
        resample_distr = self._compute_resampling_distribution(
            current_ids, current_scores
        )
        
        # Find longest common prefix
        lcp_idx = 0
        for i, (p, c) in enumerate(zip(next_ids[0], current_ids[0])):
            if p == c:
                lcp_idx += 1
            else:
                break
        
        max_resample_idx = min(lcp_idx + 1, len(current_ids[0]))
        
        # Sum over possible resampling positions
        proposal_logprob = -np.inf
        
        for i in range(max_resample_idx):
            idx_prob = resample_distr[0][i].item()
            if idx_prob == 0:
                continue
            
            idx_logprob = np.log(idx_prob)
            
            # Get suffix log probability
            suffix_ids = next_ids[:, i:]
            suffix_scores = next_scores[:, i:]
            suffix_logprob = get_seq_logprob_from_scores(
                suffix_scores,
                suffix_ids,
                self.llm.tokenizer.eos_token_id,
            ).item()
            
            # Add to total via log-sum-exp
            proposal_logprob = np.logaddexp(
                proposal_logprob,
                idx_logprob + suffix_logprob,
            )
        
        return proposal_logprob
    
    def _compute_sequence_logprob(
        self,
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> float:
        """Compute exact log probability of a sequence.
        
        Args:
            prompt_ids: Encoded prompt.
            query_ids: Sequence to compute probability for.
            
        Returns:
            Log probability.
        """
        # Use restrictor to force generation and collect log probs
        generation_config = GenerationConfig(
            max_new_tokens=query_ids.shape[1],
            num_return_sequences=1,
            do_sample=False,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        
        restrictor = _RestrictorLogitsProcessor(
            prompt_ids.size(1),
            query_ids[0],
        )
        logits_processor_list = LogitsProcessorList([restrictor])
        
        self.llm.model.generate(
            prompt_ids,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )
        
        return restrictor.result.sum().item()