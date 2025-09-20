# run_gcg.py


import os
import sys
import torch

# ensure repo local dir is importable
CUR = os.path.dirname(__file__)
if CUR not in sys.path:
    sys.path.append(CUR)

# Try to import your gcg module and classes
try:
    from gcg import GCGAttackPrompt, GCGPromptManager, GCGMultiPromptAttack
except Exception as e:
    raise RuntimeError(f"Failed to import gcg module/classes: {e}")

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# ------------------ Config (hard-coded as requested) ------------------
CAND_MODELS = [
    "gpt2",                           # usually has safetensors on HF
    "EleutherAI/pythia-70m-deduped",  # smaller alternative
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_PROMPT = "You are a helpful assistant. Answer concisely."
TARGET_TEXT = "Please refuse to answer."
CONTROL_INIT = " ! ! ! "
STEPS = 3
BATCH_SIZE = 64
TOPK = 32
# ----------------------------------------------------------------------

def load_safetensors_model(device):
    last_err = None
    for name in CAND_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            # prefer safetensors to avoid torch.load vulnerability requirement
            mdl = AutoModelForCausalLM.from_pretrained(
                name, use_safetensors=True, low_cpu_mem_usage=True
            ).to(device)
            mdl.eval()
            print(f"[INFO] Loaded model: {name}")
            return mdl, tok, name
        except Exception as e:
            last_err = e
            print(f"[WARN] Couldn't load {name} with safetensors: {repr(e)}")
    raise RuntimeError(f"No safetensors model loaded. Last error: {last_err}")

# conv_template helper: try import fastchat, else provide a minimal stub
def get_conv_template_or_stub(name="default"):
    try:
        # prefer real fastchat if available
        from fastchat.model import get_conversation_template
        conv = get_conversation_template(name)
        print("[INFO] Using fastchat.get_conversation_template")
        return conv
    except Exception:
        # provide a simple stub with minimal interface used by many AttackPrompts
        class SimpleConvTemplate:
            def __init__(self, system="You are a helpful assistant."):
                self.system = system
                self.roles = ("user", "assistant")
                self.messages = []
            def append_message(self, role, content):
                # keep API shape similar to fastchat conversation objects
                self.messages.append((role, content))
            def get_prompt(self):
                parts = []
                if self.system:
                    parts.append(self.system)
                for role, content in self.messages:
                    if role == self.roles[0]:
                        parts.append(f"User: {content}")
                    else:
                        parts.append(f"Assistant: {content}")
                return "\n".join(parts)
            # some code expects template.roles or similar attributes — we already added roles
        print("[INFO] fastchat not found; using local SimpleConvTemplate stub")
        return SimpleConvTemplate(system=BASE_PROMPT)

def build_attack_prompt_compatible(goal, target, tokenizer, conv_template, control_init):
    """
    Build GCGAttackPrompt by matching the AttackPrompt signature:
    def __init__(self, goal, target, tokenizer, conv_template, control_init=..., ...)
    We'll pass exactly these required args as keywords to be explicit.
    """
    try:
        prompt = GCGAttackPrompt(
            goal=goal,
            target=target,
            tokenizer=tokenizer,
            conv_template=conv_template,
            control_init=control_init
        )
        return prompt
    except Exception as e:
        # surface a useful error
        raise RuntimeError(f"Failed to instantiate GCGAttackPrompt with expected signature: {e}")

def main():
    print(f"[INFO] Using device: {DEVICE}")

    device = torch.device(DEVICE)

    # load model + tokenizer (safetensors-preferred)
    model, tokenizer, model_name = load_safetensors_model(device)

    # conv template (fastchat or stub)
    conv_template = get_conv_template_or_stub(name="vicuna")

    # construct GCGAttackPrompt with correct signature (goal, target, tokenizer, conv_template, control_init)
    prompt = build_attack_prompt_compatible(
        goal=BASE_PROMPT,
        target=TARGET_TEXT,
        tokenizer=tokenizer,
        conv_template=conv_template,
        control_init=CONTROL_INIT
    )

    # Prompt manager (assumes GCGPromptManager(tokenizer=..., control_str=...))
    try:
        pm = GCGPromptManager(tokenizer=tokenizer, control_str=CONTROL_INIT)
    except Exception:
        # fallback to simple construction if the class signature differs
        pm = GCGPromptManager(tokenizer=tokenizer)

    # Simple worker adapter matching calls in gcg.step: (obj, "grad", model) and (obj, "logits", model, cand, return_ids=True)
    from queue import SimpleQueue
    class SimpleWorker:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.results = SimpleQueue()
        def __call__(self, obj, mode, model, cand=None, return_ids=False):
            if mode == "grad":
                g = obj.grad(self.model)
                self.results.put(g)
            elif mode == "logits":
                # expect AttackPrompt.logits(self, model, cand, tokenizer, return_ids=True) -> (logits, ids)
                logits, ids = obj.logits(self.model, cand, self.tokenizer, return_ids=True)
                self.results.put((logits, ids))
            else:
                raise ValueError(f"Unknown mode {mode}")

    workers = [SimpleWorker(model, tokenizer, device)]

    # build attack
    attack = GCGMultiPromptAttack(
        models=[model],
        tokenizer=tokenizer,
        prompts=[[prompt]],   # single prompt in nested list as your step expects
        prompt_manager=pm,
        workers=workers,
        control_str=CONTROL_INIT,
    )

    # run a few iterations
    for step_i in range(STEPS):
        next_control, avg_loss = attack.step(
            batch_size=BATCH_SIZE,
            topk=TOPK,
            temp=1.0,
            allow_non_ascii=False,
            target_weight=1.0,
            control_weight=0.1,
            verbose=False,
            opt_only=False,
            filter_cand=True
        )
        # decode next_control token ids into string if it's token ids; if it's already str, keep
        try:
            new_control_str = tokenizer.decode(next_control, skip_special_tokens=True)
        except Exception:
            new_control_str = str(next_control)
        attack.control_str = new_control_str
        print(f"[step {step_i+1}/{STEPS}] avg_loss={avg_loss:.4f}")
        print(f"[step {step_i+1}] control_str = {repr(new_control_str)}")

    # final outputs
    print("\n=== Final Control String ===")
    print(repr(attack.control_str))

    # Build full prompt text — prefer conv_template.get_prompt if available
    try:
        # if conv_template supports append_message/get_prompt, use it to build the prompt
        if hasattr(conv_template, "append_message") and hasattr(conv_template, "get_prompt"):
            # append user message (control) and get concatenated prompt
            conv_template.append_message(conv_template.roles[0], attack.control_str)
            full_prompt = conv_template.get_prompt()
        else:
            full_prompt = BASE_PROMPT + "\n" + attack.control_str
    except Exception:
        full_prompt = BASE_PROMPT + "\n" + attack.control_str

    print("\n=== Final Full Prompt ===")
    print(full_prompt)

    # model generation sanity check
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print("\n=== Model Generation on Final Prompt ===")
        print(out_text)
    except Exception as e:
        print(f"[WARN] generation skipped: {e}")

if __name__ == "__main__":
    main()
