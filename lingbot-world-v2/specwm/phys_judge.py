"""Physics-consistency judge for the SDVG loop (10/5 meeting: 'swap the verifier for a physics reward'). A VLM (Qwen2.5-VL-7B-Instruct, local
weights hf/Qwen2.5-VL-7B-Instruct) watches a short clip and answers five yes/no violation questions, WorldModelBench-style (2502.20694):
scene change, appearance/disappearance, impossible deformation, impossible fluid/granular motion, gravity/causeless motion. Score = 5 - number of yes answers (5 = no violation). Used per chunk (previous chunk + candidate chunk decoded together, so
motion is visible) and per clip (whole video, 16 frames) for evaluation.
usage: J = PhysJudge(); score, text = J.score_frames(list_of_uint8_HxWx3, prompt)"""
import json, re, torch, numpy as np
from PIL import Image
PATH = "/home/hk28965/specwm/hf/Qwen2.5-VL-7B-Instruct"
SYSTEM = ("You are a strict physics reviewer for generated videos. The frames are consecutive samples of ONE short video, in temporal order (frame 1 = start). "
          "Camera motion itself is NOT a violation.")
QUESTIONS = ("Answer each question with yes or no:\n"
             "Q1 Does the video move to a DIFFERENT scene or location than frame 1 (not just the camera turning)?\n"
             "Q2 Do any people, animals, vehicles or objects APPEAR that were not present in frame 1, or does anything present in frame 1 DISAPPEAR without leaving the view?\n"
             "Q3 Does any object change its identity, melt, warp or deform in a way that a real object could not?\n"
             "Q4 Does water, sand, liquid or any moving material move in a physically impossible way (flow uphill, freeze mid-air, appear from nothing)?\n"
             "Q5 Does anything float, fall, or move in a way that ignores gravity or has no cause?\n"
             "Reply with JSON only: {\"Q1\": \"yes\"|\"no\", \"Q2\": ..., \"Q3\": ..., \"Q4\": ..., \"Q5\": ..., \"notes\": \"<=40 words\"}")
class PhysJudge:
    def __init__(self, device="cuda:0", dtype=torch.bfloat16, max_side=640):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(PATH, torch_dtype=dtype, device_map={"": device}, attn_implementation="sdpa").eval()
        self.p = AutoProcessor.from_pretrained(PATH); self.dev = device; self.max_side = max_side
    @staticmethod
    def pick(frames, n):
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int); return [frames[i] for i in idx]
    @torch.no_grad()
    def score_frames(self, frames, context="", n=12, max_new=160):
        """frames: list of uint8 HxWx3 arrays (temporal order). returns (score float 1-5 or None, raw text)"""
        fr = self.pick(frames, n); ims = []
        for f in fr:
            im = Image.fromarray(f); s = self.max_side / max(im.size)
            if s < 1: im = im.resize((int(im.width * s) // 28 * 28, int(im.height * s) // 28 * 28), Image.BICUBIC)
            ims.append(im)
        user = ("Scene description: %s\n" % context if context else "") + QUESTIONS
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user}]}]
        text = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.p(text=[text], videos=[ims], return_tensors="pt").to(self.dev)
        out = self.m.generate(**inp, max_new_tokens=max_new, do_sample=False)
        ans = self.p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        qs = re.findall(r"\"Q\d\"\s*:\s*\"(yes|no)\"", ans); return (5.0 - sum(q == "yes" for q in qs)) if len(qs) == 5 else None, ans
