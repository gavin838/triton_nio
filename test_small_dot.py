import numpy as np
import torch
import triton
import triton.language as tl
import pytest

device = "cuda"


@triton.jit
def _triton_dot_kernel(
    # Pointers to matrices
    A,
    B,
    C,
    # Matrix dimensions
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    As = A + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    Bs = B + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn
    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(As)
        b = tl.load(Bs)
        c += tl.dot(a, b)
        As += stride_ak * BLOCK_K
        Bs += stride_bk * BLOCK_K
    Cs = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(Cs, c)


@triton.jit
def _triton_dot3d_kernel(
        # Pointers to matrices
        A, B, C,
        # Matrix dimensions
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K: tl.constexpr, K_SPLIT: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    As = A + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    Bs = B + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(As)
        a = tl.reshape(a, (BLOCK_M, K_SPLIT, BLOCK_K // K_SPLIT)).trans(1, 0, 2)
        b = tl.load(Bs)
        b = tl.reshape(b, (K_SPLIT, BLOCK_K // K_SPLIT, BLOCK_N))
        c = tl.dot(a, b)
        c = tl.sum(c, axis=0)
        As += stride_ak * BLOCK_K
        Bs += stride_bk * BLOCK_K
    Cs = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(Cs, c)


def gemm_forward(out, a, b, dot3d):
    M, K = a.shape
    K, N = b.shape
    BLOCK_K = 512

    kwargs = [
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
        BLOCK_K,
        K,
    ]
    if dot3d:
        kwargs += [8]

    if dot3d:
        _triton_dot3d_kernel[(1, 1)](*kwargs)
    else:
        _triton_dot_kernel[(1, 1)](*kwargs)


def get_shapes():
    shapes = [(i, 32, 2048) for i in (1, 2, 4, 8)]
    return shapes


name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
}


def gen_input(M, N, ty_name, needTrans, seed, device):
    d_type = name_to_tl_types[ty_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if ty_name == 'int8':
        if needTrans:
            raw_data = torch.randint(-10, 10, (N, M), dtype=torch.int8, device=device).T
        else:
            raw_data = torch.randint(-10, 10, (M, N), dtype=torch.int8, device=device)

        return raw_data, raw_data.to(torch.half)


@pytest.mark.parametrize('m, n, k', get_shapes())
def test_gemm(m, n, k):
    torch.random.manual_seed(0)
    with torch.no_grad():
        a, _ = gen_input(m, k, 'int8', False, 1, device=device)
        b, _ = gen_input(k, n, 'int8', True, 2, device=device)
        #for i in range(m):
        #    for j in range(k):
        #        a[i, j] = 1 if j == 0 else 0
        #for i in range(n):
        #    for j in range(k):
        #        b[j, i] = (i) % 127

        out_torch = torch.matmul(a.to(torch.int32).cpu(), b.to(torch.int32).cpu())
        out_triton = torch.empty([a.shape[0], b.shape[1]], dtype=torch.int32, device=a.device)
        gemm_forward(out_triton, a, b, dot3d=False)

        print("triton: ", out_triton.cpu().numpy())
        print("ref: ", out_torch.numpy())

        diff = ~np.isclose(out_triton.cpu().numpy(), out_torch.numpy(), rtol=1e-2)
        assert diff.sum() < 10, f"m={m}, n={n}, k={k}"
