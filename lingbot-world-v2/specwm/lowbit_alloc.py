"""kvstore2 를 쓸 때 bf16 KV 캐시를 실제로 할당하지 않는다.

문제: kvstore2.install() 은 기존 bf16 캐시를 shape 읽는 용도로만 쓰고 곧바로
torch.empty(0) 으로 교체한다. 그런데 그 bf16 텐서는 이미 할당된 뒤다.
창 72 기준 85 GiB 를 잡았다 버리는 셈이라, 압축 후 크기가 22 GiB 여도 OOM 이 난다.
(실측: 잡 965441 의 W72_kv4 arm 이 92.94 GiB 사용 중 OOM)

해결: 캐시를 만들 때 k/v 를 빈 텐서로 두고 의도한 shape 만 'kv_shape' 로 기록한다.
kvstore2 는 numel()==0 이면 kv_shape 에서 T 를 읽는다(kvstore2 쪽 1줄 수정 필요).

주의: kvstore2 를 설치하지 않는 경로에서는 절대 쓰면 안 된다.
wan/distributed/sequence_parallel.py 가 실제 bf16 텐서에 write 하기 때문이다.
"""
import torch


def patch(pipe, verbose=True):
    """pipe._initialize_self_kv_cache 를 지연 할당 버전으로 교체하고 원본을 돌려준다."""
    orig = pipe._initialize_self_kv_cache

    def lean(num_layers, shape, dtype, device):
        out = []
        for _ in range(num_layers):
            out.append({
                "k": torch.empty(0, dtype=dtype, device=device),
                "v": torch.empty(0, dtype=dtype, device=device),
                "kv_shape": tuple(int(x) for x in shape),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        return out

    pipe._initialize_self_kv_cache = lean
    if verbose:
        el = 2 if str(getattr(pipe, "param_dtype", "bfloat16")).endswith("16") else 4
        print("[lowbit_alloc] bf16 선할당 우회 활성화", flush=True)
    return orig


def unpatch(pipe, orig):
    pipe._initialize_self_kv_cache = orig


def would_save_GiB(shape, num_layers, bytes_per=2):
    """이 패치가 아끼는 순간 할당량(GiB). k 와 v 두 개."""
    n = 1
    for x in shape:
        n *= int(x)
    return 2 * n * num_layers * bytes_per / 2 ** 30
