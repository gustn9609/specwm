#!/bin/bash
S=$SCRATCH; D=$S/aria; mkdir -p $D; cd $D
true
echo "== pip install projectaria_tools =="; pip install -q projectaria_tools 2>&1 | tail -2; python -c "import projectaria_tools; from projectaria_tools.core import data_provider; print('projectaria_tools OK', projectaria_tools.__version__)" 2>&1 | tail -1
for f in loc1_script2_seq1_rec1_10s_sample.zip loc1_script2_seq1_rec2_10s_sample.zip; do [ -f $f ] || curl -sL -o $f "https://www.projectaria.com/async/sample/download/?bucket=aea&filename=$f"; ls -la $f | awk '{print $5, $9}'; done
for f in *.zip; do d=${f%.zip}; mkdir -p $d; unzip -q -o $f -d $d 2>/dev/null; echo "== $d"; find $d -maxdepth 3 | head -20; done
