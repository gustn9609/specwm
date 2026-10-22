#!/bin/bash
S=$SCRATCH
echo "=== 궤적 동일한가 (00 vs 05) ==="
python3 -c "
import numpy as np
a=np.load('$S/rt2_action_00/poses.npy'); b=np.load('$S/rt2_action_05/poses.npy')
print('  poses 동일?', np.allclose(a,b), a.shape, b.shape)
i0=np.load('$S/rt2_idx.npy'); i5=np.load('$S/revisit_rt_idx_05.npy')
print('  idx 동일?', np.array_equal(i0,i5))"
cat > $S/geos05.slurm << 'SEOF'
#!/bin/bash
true
source $SCRATCH/venv/bin/activate
cd $SCRATCH && python geo_s05.py
SEOF
sbatch $S/geos05.slurm | tail -1
