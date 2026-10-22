#!/bin/bash
cd $SCRATCH; echo "== projectaria_tools 설치 여부 =="; module load gcc cuda python3 >/dev/null 2>&1; source $SCRATCH/venv/bin/activate; python -c "import projectaria_tools; print('installed', projectaria_tools.__version__)" 2>&1 | tail -1
echo "== AEA 공개 샘플 접근 테스트 =="; curl -sIL https://www.projectaria.com/datasets/aea/ | grep -iE "^HTTP" | tail -1
