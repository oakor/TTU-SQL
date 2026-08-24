#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPO (Negative Preference Optimization) Trainer
 open-unlearning 
 token  (forget_mask)
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from typing import Dict, Optional


class NPOTrainer(Trainer):
    """
    NPO 
    
    :
    - forget loss:  DPO loss ( forget  negative samples)
    - retain loss:  NLL loss
    -  token  (forget_mask)
    """
    
    def __init__(
        self,
        model,
        ref_model,
        beta=0.1,
        alpha=1.0, 
        gamma=1.0,
        debug=True,
        *args,
        **kwargs
    ):
        """
        Args:
            model: 
            ref_model:  ()
            beta: DPO 
            alpha: retain loss 
            gamma: forget loss 
            debug: 
        """
        super().__init__(model=model, *args, **kwargs)
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.debug = debug
        self._debug_count = 0  #  debug 
    
    def compute_batch_nll(self, model, inputs, forget_mask=None):
        """
         (NLL)
         token  (forget_mask)
        
        Args:
            model: 
            inputs: 
            forget_mask: shape (batch_size, seq_len)，1  loss
                         None， loss
        
        Returns:
            loss: shape (batch_size,) 
            outputs: 
        """
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift for next token prediction
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        
        #  token ， reduction
        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_losses = loss_function(
            shifted_logits.transpose(-1, -2), 
            shifted_labels
        )  # shape: (batch_size, seq_len-1)
        
        if forget_mask is not None:
            #  forget_mask， mask=1  loss
            # forget_mask  shift
            shifted_forget_mask = forget_mask[..., 1:].contiguous().float()
            
            #  mask=0  loss  0
            masked_token_losses = token_losses * shifted_forget_mask
            
            # 
            loss = masked_token_losses.sum(dim=-1)
            
            # Debug 
            if self.debug and self._debug_count < 3:
                batch_size = token_losses.shape[0]
                for i in range(min(2, batch_size)):
                    total_tokens = (shifted_labels[i] != -100).sum().item()
                    forget_tokens = shifted_forget_mask[i].sum().item()
                    print(f"[DEBUG] Sample {i}: total_valid_tokens={total_tokens}, "
                          f"forget_tokens={int(forget_tokens)}, "
                          f"forget_ratio={forget_tokens/max(total_tokens,1)*100:.1f}%")
                    print(f"[DEBUG] Sample {i}: masked_loss={masked_token_losses[i].sum().item():.4f}, "
                          f"full_loss={token_losses[i].sum().item():.4f}")
        else:
            # 
            loss = token_losses.sum(dim=-1)
        
        return loss, outputs
    
    def compute_dpo_loss(self, model, ref_model, lose_inputs, beta, forget_mask=None):
        """
         DPO  ( negative samples，)
        
         NPO， forget samples  lose_inputs
        win_inputs  None ( preferred samples)
        
        DPO loss for unlearning:
        loss = -2/β * log_sigmoid(β * (0 - lose_log_ratio))
             = -2/β * log_sigmoid(-β * lose_log_ratio)
        
         forget samples 
        
        Args:
            forget_mask: token  mask， mask=1  loss
        """
        # （ forget_mask）
        model_inputs = {
            "input_ids": lose_inputs["input_ids"],
            "attention_mask": lose_inputs["attention_mask"],
            "labels": lose_inputs["labels"],
        }
        
        # （）
        lose_loss, lose_outputs = self.compute_batch_nll(model, model_inputs, forget_mask)
        
        # （）
        with torch.no_grad():
            lose_ref_loss, _ = self.compute_batch_nll(ref_model, model_inputs, forget_mask)
        
        lose_ref_loss = lose_ref_loss.detach()
        
        # Log ratio: log(π_θ / π_ref)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
        
        # DPO loss
        loss = -2 / beta * F.logsigmoid(-beta * lose_log_ratio).mean()
        
        # Debug 
        if self.debug and self._debug_count < 3:
            print(f"[DEBUG] DPO Loss:")
            print(f"  - lose_loss (current): {lose_loss.mean().item():.4f}")
            print(f"  - lose_ref_loss (ref): {lose_ref_loss.mean().item():.4f}")
            print(f"  - log_ratio: {lose_log_ratio.mean().item():.4f}")
            print(f"  - DPO loss: {loss.item():.4f}")
        
        return loss, lose_outputs
    
    def compute_retain_loss(self, model, retain_inputs):
        """
         retain loss ()
        """
        outputs = model(**retain_inputs)
        return outputs.loss, outputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
         = gamma * forget_loss + alpha * retain_loss
        
        inputs :
        {
            "forget": {..., "forget_mask": ...},  # forget samples with token-level mask
            "retain": {...}   # retain samples  
        }
        
        Args:
            model: 
            inputs: 
            return_outputs: 
            num_items_in_batch: batch  ( transformers ，)
        """
        # 1.  forget loss ( DPO， forget_mask)
        forget_inputs = inputs["forget"]
        forget_mask = forget_inputs.get("forget_mask", None)
        
        if self.debug and self._debug_count < 3:
            print(f"\n{'='*60}")
            print(f"[DEBUG] Step {self.state.global_step}: Computing losses")
            if forget_mask is not None:
                print(f"[DEBUG] forget_mask shape: {forget_mask.shape}")
                print(f"[DEBUG] forget_mask sum per sample: {forget_mask.sum(dim=-1).tolist()[:4]}...")  # 4
        
        forget_loss, forget_outputs = self.compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            lose_inputs=forget_inputs,
            beta=self.beta,
            forget_mask=forget_mask
        )
        
        # 2.  retain loss ( NLL)
        retain_inputs = inputs["retain"]
        retain_inputs_filtered = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss, retain_outputs = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs_filtered
        )
        
        # 3. 
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        
        # Debug 
        if self.debug and self._debug_count < 3:
            print(f"[DEBUG] Final: forget_loss={forget_loss.item():.4f}, "
                  f"retain_loss={retain_loss.item():.4f}, total={loss.item():.4f}")
            print(f"{'='*60}\n")
            self._debug_count += 1
        
        # 
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "forget_loss": forget_loss.item(),
                "retain_loss": retain_loss.item(),
                "total_loss": loss.item(),
            })
        
        return (loss, forget_outputs) if return_outputs else loss
