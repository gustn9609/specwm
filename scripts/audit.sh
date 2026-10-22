#!/bin/bash
S=$SCRATCH
echo "=== 프롬프트 파일 ==="
for f in $S/p_*.txt; do echo "--- $(basename $f)"; cat "$f"; echo; done
echo "=== v2 examples 이미지 ==="
ls -la $S/lingbot-world-v2/examples/*/image.jpg 2>/dev/null | awk '{print $5, $9}'
echo "=== v1(Base) examples 이미지 ==="
ls -la $S/lingbot-world/examples/*/image.jpg 2>/dev/null | awk '{print $5, $9}'
echo "=== 두 저장소 이미지 동일한가 (md5) ==="
for sc in 00 01 05; do
  a=$(md5sum $S/lingbot-world-v2/examples/$sc/image.jpg 2>/dev/null | cut -d' ' -f1)
  b=$(md5sum $S/lingbot-world/examples/$sc/image.jpg 2>/dev/null | cut -d' ' -f1)
  echo "  $sc  v2=$a  v1=$b  $([ "$a" = "$b" ] && echo 동일 || echo 다름/없음)"
done
