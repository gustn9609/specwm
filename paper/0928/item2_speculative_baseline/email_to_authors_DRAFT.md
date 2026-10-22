**To:** Valentin De Bortoli <vdebortoli@google.com>

This is the only address printed in the paper ("Correspondence to: Valentin De Bortoli"). The literature agent found it in the ICML PDF and in arXiv v1 and v2, and I re-checked it in the v2 HTML: https://arxiv.org/html/2501.05370v2.

**Cc (optional):** Alexandre Galashov (co-first author). His address is not printed in the paper (UNVERIFIED). Please do not guess one; add it only if you find it confirmed.

**Subject:** Reproducing "Accelerated Diffusion Models via Speculative Sampling" (ICML 2025): code and hyper-parameters

---

Dear Dr. De Bortoli,

I am Hyunsoo, a student at The University of Texas at Austin working with Prof. Guandao [surname — please confirm] and Prof. Diana [surname — please confirm]. I enjoyed your ICML 2025 paper on speculative sampling for diffusion models, and we would like to use it as the reference baseline in our project.

Our setting is autoregressive video world models. The draft is a distilled 14B block-causal video diffusion model (LingBot-World "Fast"). It generates a video chunk by chunk (16×3×58×106 latents per chunk) using four x0-prediction-and-renoise steps. The target is Wan2.2-I2V-A14B, a mixture of two 14B experts (bidirectional attention, 70 steps, CFG 5). We implemented your reflection maximal coupling (Algorithm 4) for the three stochastic transitions of each chunk. We couple each latent position separately. The target transition is one CFG forward of Wan2.2, with the previously generated frames re-noised to the same noise level. In this setting the draft and target transition means differ a lot, so before drawing any conclusions we would first like to reproduce your CIFAR-10 result exactly, to validate our implementation.

Could you help us with the following?
1. Is code available, or could it be released? If not, is there a public implementation you consider faithful?
2. For Table 1 (CIFAR-10, 100-step target, frozen target draft):
   - the window length L;
   - the pretrained network and its noise schedule;
   - the number of samples and the reference statistics used for FID;
   - whether churn ε and temperature τ are the only settings that change between rows.
3. How target NFE is counted when a rejection happens in the middle of a window. We read it as one batched target call per window.
4. Do you have any experience with a few-step distilled draft whose x0 predictions are sharp while the target's are posterior means? For example, did you try coupling per coordinate block, or a different stochastic sampler for the target?

If the block-verification follow-up (Soen et al., 2026) has code, a pointer to it would also help.

Thank you very much for your time. We are happy to share our implementation.

Best regards,
Hyunsoo [last name]
[program], The University of Texas at Austin
Advisors: [please fill in]
[UT email address]

---
*Not sent: I cannot send email. Please fill in the bracketed fields (advisor names were not confirmed in any file I could read) and decide who should be cc'd.*