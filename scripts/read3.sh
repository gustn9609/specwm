#!/bin/bash
S=$SCRATCH
echo "===== hybrid_refine.py CLI / 핵심 루프 ====="; grep -nE "add_argument|def |sigma|low_noise|make_x0|sch\.|timesteps|latent\.pt|np.save|imageio" $S/lingbot-world/specwm/hybrid_refine.py | head -40
echo; echo "===== FlowUniPC sigmas 속성 ====="; grep -nE "self.sigmas|def set_timesteps|self.timesteps =" $S/lingbot-world/wan/utils/fm_solvers_unipc.py | head -8
echo; echo "===== rt2 idx 05 ====="; python3 -c "
import numpy as np; i=np.load('$S/revisit_rt_idx_05.npy'); P=np.load('$S/poses_05.npy'); print('len', len(i), 'first', i[0], 'last', i[-1], 'poses', P.shape)"
echo; echo "===== v2 binding node sigmas ====="; grep -n "def node_sigmas" -A4 $S/lingbot-world-v2/specwm/binding_v2.py
