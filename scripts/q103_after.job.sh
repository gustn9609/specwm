true
# usage: WAITLOG=logs/x.out WAITPAT=Q100_DONE NEXT="cmd..." — 로그에 패턴이 뜨면 NEXT 실행
for i in $(seq 1 2000); do grep -q "$WAITPAT" $SCRATCH/$WAITLOG 2>/dev/null && break; sleep 60; done
eval "$NEXT"
