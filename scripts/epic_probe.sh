#!/bin/bash
S=$SCRATCH; D=$S/epic; mkdir -p $D; cd $D
[ -f EPIC_100_train.csv ] || curl -sL -o EPIC_100_train.csv https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv
[ -f EPIC_100_validation.csv ] || curl -sL -o EPIC_100_validation.csv https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_validation.csv
ls -la EPIC_100_*.csv | awk '{print $5, $9}'; head -1 EPIC_100_train.csv | cut -c1-300
echo "--- P28_101 narrations (train+val) ---"
grep -h "^P28_101_\|,P28_101," EPIC_100_train.csv EPIC_100_validation.csv | wc -l
echo "--- EPIC-Fields example json ---"
[ -f P28_101.json ] || curl -sL -o P28_101.json https://raw.githubusercontent.com/epic-kitchens/epic-fields-code/main/example_data/P28_101.json
ls -la P28_101.json | awk '{print $5, $9}'; python3 -c "
import json; d=json.load(open('P28_101.json')); print('keys', list(d.keys())); print('camera', d['camera']); im=d['images']; ks=sorted(im.keys()); print('n frames', len(ks), ks[0], ks[-1]); print('example', ks[0], im[ks[0]])"
echo "--- video HEAD ---"
curl -sI -L https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P28/videos/P28_101.MP4 | grep -iE "HTTP/|content-length|accept-ranges|content-type" | head -6
