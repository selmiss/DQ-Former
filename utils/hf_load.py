import os
import json
import glob
from typing import Dict, List, Optional, Tuple

def _hf_cache_dir() -> str:
    env = os.environ
    if "HUGGINGFACE_HUB_CACHE" in env and env["HUGGINGFACE_HUB_CACHE"]:
        return env["HUGGINGFACE_HUB_CACHE"]
    if "HF_HOME" in env and env["HF_HOME"]:
        return os.path.join(env["HF_HOME"], "hub")
    # 一些历史环境可能把模型放在 TRANSFORMERS_CACHE，下行作为兜底之一
    if "TRANSFORMERS_CACHE" in env and env["TRANSFORMERS_CACHE"]:
        return env["TRANSFORMERS_CACHE"]
    return os.path.expanduser("~/.cache/huggingface/hub")

def _repo_cache_dir(repo_id: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or _hf_cache_dir()
    if "/" not in repo_id:
        # 例如 "gpt2" 这种别名会被展开为 models--None--gpt2；大多数情况下还是建议传完整 org/repo
        org, name = "None", repo_id
    else:
        org, name = repo_id.split("/", 1)
    return os.path.join(cache_dir, f"models--{org}--{name}")

def _resolve_snapshot_dir(repo_dir: str, prefer_ref: str = "main") -> Optional[str]:
    """优先根据 refs/<prefer_ref> 解析具体 revision；否则选最近修改的 snapshot。"""
    refs_dir = os.path.join(repo_dir, "refs")
    snap_dir = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(refs_dir):
        ref_file = os.path.join(refs_dir, prefer_ref)
        if os.path.isfile(ref_file):
            with open(ref_file, "r") as f:
                rev = f.read().strip()
            candidate = os.path.join(snap_dir, rev)
            if os.path.isdir(candidate):
                return candidate
    # 回退：选择最近修改的 snapshot 目录
    if os.path.isdir(snap_dir):
        snaps = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)]
        snaps = [d for d in snaps if os.path.isdir(d)]
        if snaps:
            snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return snaps[0]
    return None

def _files_from_index(index_path: str) -> List[str]:
    """从 *index.json 里解析分片文件名（唯一列表，保持字母序）。"""
    with open(index_path, "r") as f:
        idx = json.load(f)
    # Transformers 的 index 里通常是 {"weight_map": {"param_name": "shard_file", ...}}
    wm = idx.get("weight_map", {})
    files = sorted(set(wm.values()))
    base = os.path.dirname(index_path)
    return [os.path.join(base, fn) for fn in files]

def _find_weight_files(snapshot_dir: str) -> Tuple[List[str], Optional[str]]:
    """
    返回 (weights, index_path)。若存在 *index.json 则以之为准，否则用通配回退。
    """
    # 1) 优先 safetensors 索引
    st_idx = glob.glob(os.path.join(snapshot_dir, "*.safetensors.index.json"))
    if st_idx:
        index_path = sorted(st_idx)[0]
        return _files_from_index(index_path), index_path
    # 2) 再看 pytorch bin 索引
    pt_idx = glob.glob(os.path.join(snapshot_dir, "*.bin.index.json"))
    if pt_idx:
        index_path = sorted(pt_idx)[0]
        return _files_from_index(index_path), index_path

    # 3) 没有索引：用常见文件名匹配（尽量排除非权重的 .bin）
    candidates = []
    patterns = [
        "model*.safetensors", "pytorch_model*.safetensors", "*.safetensors",
        "pytorch_model*.bin", "model*.bin"
    ]
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(snapshot_dir, pat)))

    # 过滤明显不是权重的 .bin（如 trainer_state.json 等不是 .bin，已被模式排除）
    # 若同名 safetensors 和 bin 同时存在，优先 safetensors
    st = [p for p in candidates if p.endswith(".safetensors")]
    if st:
        return sorted(set(st)), None
    pt = [p for p in candidates if p.endswith(".bin")]
    return sorted(set(pt)), None

def locate_hf_cached_weights(repo_id: str, cache_dir: Optional[str] = None) -> Dict:
    """
    根据 repo_id 在本地 HF 缓存定位权重文件。
    返回 dict:
      {
        "cache_dir": <hub 目录>,
        "repo_dir": <models--ORG--REPO>,
        "snapshot_dir": <具体 revision 目录>,
        "weight_files": [<绝对路径>...],
        "index_file": <index.json 或 None>
      }
    若找不到会抛出 RuntimeError。
    """
    repo_dir = _repo_cache_dir(repo_id, cache_dir)
    if not os.path.isdir(repo_dir):
        raise RuntimeError(f"本地没有缓存 {repo_id!r}（未下载或缓存目录不同）: {repo_dir}")
    snapshot_dir = _resolve_snapshot_dir(repo_dir)
    if not snapshot_dir:
        raise RuntimeError(f"未找到 {repo_id!r} 的 snapshot 目录：{os.path.join(repo_dir, 'snapshots')}")
    weights, index_file = _find_weight_files(snapshot_dir)
    if not weights:
        raise RuntimeError(f"在 {snapshot_dir} 未发现权重文件（*.safetensors / *.bin）")
    return {
        "cache_dir": _hf_cache_dir(),
        "repo_dir": repo_dir,
        "snapshot_dir": snapshot_dir,
        "weight_files": weights,
        "index_file": index_file,
    }

def load_state_dict_from_cache(repo_id: str, map_location: str = "cpu") -> Tuple[Dict, Dict]:
    """
    直接从缓存加载为合并后的 state_dict（可能占较多内存）。
    返回 (state_dict, info_dict)，其中 info_dict 为 locate_hf_cached_weights 的返回。
    """
    info = locate_hf_cached_weights(repo_id)
    paths = info["weight_files"]
    sd: Dict = {}
    if paths[0].endswith(".safetensors"):
        import safetensors.torch as st
        for p in paths:
            shard = st.load_file(p, device=map_location if map_location != "cpu" else "cpu")
            sd.update(shard)
    else:
        import torch
        for p in paths:
            shard = torch.load(p, map_location=map_location)
            # torch.load 可能返回 {"state_dict": ...} 结构，做个兼容
            if isinstance(shard, dict) and "state_dict" in shard and all(
                isinstance(k, str) for k in shard["state_dict"].keys()
            ):
                shard = shard["state_dict"]
            sd.update(shard)
    return sd, info
