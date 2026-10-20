import os, json, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import trange
import fire

# -----------------------------
# Global flags
# -----------------------------
STRIP_LEXICAL = False

# -----------------------------
# Lexicon I/O
# -----------------------------
def _read_list(path: str) -> List[str]:
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                tok, *_ = line.split("\t")
                toks.append(tok)
            else:
                toks.append(line)
    return toks

def load_lexicon(
    lex_dir: str,
    fallback_sg=None, fallback_pl=None, fallback_verbs=None, fallback_dets=None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    sg_path = os.path.join(lex_dir, "sg_nouns.txt")
    pl_path = os.path.join(lex_dir, "pl_nouns.txt")
    v_path  = os.path.join(lex_dir, "verbs.txt")
    d_path  = os.path.join(lex_dir, "dets.txt")

    sg = _read_list(sg_path) if os.path.exists(sg_path) else (fallback_sg or [])
    pl = _read_list(pl_path) if os.path.exists(pl_path) else (fallback_pl or [])
    vb = _read_list(v_path)  if os.path.exists(v_path)  else (fallback_verbs or [])
    dt = _read_list(d_path)  if os.path.exists(d_path)  else (fallback_dets or [])

    if not sg or not pl or not vb or not dt:
        raise FileNotFoundError(
            "Missing lexicon files. Expected: "
            f"{sg_path}, {pl_path}, {v_path}, {d_path}"
        )
    return sg, pl, vb, dt

# Minimal defaults
DEFAULT_SG_NOUNS = ["cat","dog","book","man","girl","house","tree","car"]
DEFAULT_PL_NOUNS = ["cats","dogs","books","men","girls","houses","trees","cars"]
DEFAULT_VERBS    = ["see","chase","find","like","watch","admire"]
DEFAULT_DETS     = ["the"]

# -----------------------------
# MG data structures
# -----------------------------
@dataclass
class DPNode:
    det: str
    noun: str
    num: str            # 'sg' | 'pl': Interpretable Number feature (i-Num)
    wh_minus: bool      # True means [-wh]: Non-wh-phrase (for A-bar movement goal)
    trace: bool = False

@dataclass
class VLex:
    v: str
    # sel_order: Tuple[str, str] = ("d","d") # Not used, removed

@dataclass
class TVLex:
    epp: bool = True
    u_num: Optional[str] = None # Uninterpretable Number feature (u-Num)

@dataclass
class CLex:
    wh_plus: bool = False # True means [+wh]: Triggers wh-movement

@dataclass
class VPNode:
    verb: VLex
    obj: DPNode
    subj: DPNode

@dataclass
class TPNode:
    T: TVLex
    VP: VPNode
    spec: Optional[DPNode] = None

@dataclass
class CPNode:
    C: CLex
    TP: TPNode
    spec: Optional[DPNode] = None

# -----------------------------
# MG operations
# -----------------------------
def agree_T(T: TVLex, candidates: List[DPNode]):
    """Agree: T's u-Num feature is valued by the subject's i-Num feature."""
    if not candidates:
        return
    goal = candidates[-1]  # subject
    T.u_num = goal.num

def epp_move(tp: TPNode):
    """Move: Subject moves to Spec-TP to satisfy T's EPP feature."""
    if not tp.T.epp or tp.spec is not None:
        return
    subj = tp.VP.subj
    # Create a copy of DP at Spec-TP (Non-trace position)
    tp.spec = DPNode(det=subj.det, noun=subj.noun, num=subj.num,
                     wh_minus=subj.wh_minus, trace=False)
    # Mark the original position as a trace
    tp.VP.subj.trace = True

def wh_move(cp: CPNode):
    """Move: A DP with [-wh] feature moves to Spec-CP if C has [+wh] feature."""
    if not cp.C.wh_plus:
        return
    
    # Candidates ordered by proximity/Minimal Link Condition (MLC)
    cand: List[Tuple[str, DPNode]] = []
    if cp.TP.spec and not cp.TP.spec.wh_minus: # Check if it's the moved subject's copy
        # If Spec-TP is a DP, it must be the one that moved from Subj position
        # Only DP that moved to Spec-TP is considered for further movement if it has [-wh]
        if not cp.TP.spec.trace and cp.TP.spec.wh_minus:
             cand.append(("SpecTP", cp.TP.spec))
    
    # Check original/other positions (Obj is closer to T/C than original Subj trace)
    if cp.TP.VP.obj and cp.TP.VP.obj.wh_minus:
        cand.append(("Obj", cp.TP.VP.obj))
    if cp.TP.VP.subj and cp.TP.VP.subj.wh_minus:
        cand.append(("SubjTrace" if cp.TP.VP.subj.trace else "Subj", cp.TP.VP.subj))

    if not cand:
        return
        
    pos, goal = cand[0] # Select the highest/closest candidate
    
    # Create a copy of DP at Spec-CP
    # The feature [-wh] is assumed to be checked/erased upon movement
    cp.spec = DPNode(det=goal.det, noun=goal.noun, num=goal.num,
                     wh_minus=False, trace=False)
                     
    # Mark the original position (or Spec-TP position) as a trace
    if pos == "SpecTP": cp.TP.spec.trace = True
    elif pos == "Obj": cp.TP.VP.obj.trace = True
    else: cp.TP.VP.subj.trace = True

# -----------------------------
# Linearization helpers
# -----------------------------
def tok_V(v: str) -> List[str]:
    return ["V"] if STRIP_LEXICAL else [f"V({v})"]

def tok_D(d: str) -> List[str]:
    return ["D"] if STRIP_LEXICAL else [f"D({d})"]

def tok_N(n: str) -> List[str]:
    return ["N"] if STRIP_LEXICAL else [f"N({n})"]

def lin_DP(dp: DPNode, enable_merge: bool = True) -> List[str]:
    # Trace linearizations
    if dp.trace and dp.wh_minus:
        return ["TR[-wh]"]
    if dp.trace:
        return ["TR[DP]"]
    
    # Base DP structure
    dp_label = f"DP[Num:{dp.num}]"
    wh_label = ["[-wh]"] if dp.wh_minus else []

    if enable_merge:
        # [ DP[Num:Num] [ D ] [ N ] ] [-wh] (Simplified Merge structure)
        out = ["[" , dp_label, "[" ] + tok_D(dp.det) + ["]", "[" ] + tok_N(dp.noun) + ["]", "]"]
    else:
        # D N DP[Num:Num] [-wh]
        out = tok_D(dp.det) + tok_N(dp.noun) + [dp_label]
        
    return out + wh_label

def lin_VP(vp: VPNode, enable_merge: bool = True, pos_v_obj_subj: bool = True) -> List[str]:
    v_tok = tok_V(vp.verb.v)
    obj_tok = lin_DP(vp.obj, enable_merge)
    subj_tok = lin_DP(vp.subj, enable_merge)

    if pos_v_obj_subj:
        # Default/SOV-like: V + Obj + Subj
        order = obj_tok + subj_tok
        vp_content = v_tok + ["["] + obj_tok + ["]", "["] + subj_tok + ["]"]
    else:
        # SVO-like: V + Subj + Obj
        order = subj_tok + obj_tok
        vp_content = v_tok + ["["] + subj_tok + ["]", "["] + obj_tok + ["]"]

    if enable_merge:
        # [ VP V [ Obj ] [ Subj ] ] (or V [ Subj ] [ Obj ])
        return ["[", "VP"] + vp_content + ["]"]

    else:
        # V Obj Subj (or V Subj Obj)
        return v_tok + order

def lin_T(T: TVLex, enable_merge: bool = True) -> List[str]:
    t_num = T.u_num if T.u_num else 'u'
    t_epp_label = "+EPP," if T.epp else ""
    t_head = f"T({t_epp_label}uNum:{t_num})"
        
    if enable_merge:
        # [ T_head ] (Head)
        return ["[", t_head, "]"]
    return [t_head]

def lin_TP(tp: TPNode, enable_merge: bool = True, pos_v_obj_subj: bool = True) -> List[str]:
    
    t_tok = lin_T(tp.T, enable_merge)
    vp_tok = lin_VP(tp.VP, enable_merge, pos_v_obj_subj)
    
    # Spec-TP Linearization
    spec_tok = []
    if tp.spec:
        spec_tok = ["["] + lin_DP(tp.spec, enable_merge) + ["]"]
    
    if enable_merge:
        # [ TP (Spec-TP) [ T ] [ VP ] ]
        return ["[", "TP"] + spec_tok + t_tok + ["["] + vp_tok + ["]", "]"]
    
    # T VP (Spec-TP is handled by epp_move and its linearization in lin_DP if not trace)
    # Simplification: Spec-TP is linearized before T if it exists
    return spec_tok + t_tok + vp_tok

def lin_C(C: CLex, enable_merge: bool = True) -> List[str]:
    c_head = "C"
    if C.wh_plus:
        c_head = "C(+wh)"
    
    if enable_merge:
        return ["[", c_head, "]"]
    return [c_head]


def lin_CP(cp: CPNode, enable_merge: bool = True, pos_v_obj_subj: bool = True) -> List[str]:

    c_tok = lin_C(cp.C, enable_merge)
    tp_tok = lin_TP(cp.TP, enable_merge, pos_v_obj_subj)
    
    # Spec-CP Linearization (Moved wh-phrase)
    spec_tok = []
    if cp.spec:
        # Assuming the moved constituent is linearized first in Spec-CP
        spec_tok = ["MOV[+wh]", "[" ] + lin_DP(cp.spec, enable_merge) + ["]"]
    
    if enable_merge:
        # [ CP Spec-CP [ C ] [ TP ] ]
        return ["[", "CP"] + spec_tok + c_tok + ["["] + tp_tok + ["]", "]"]
    else:
        # Spec-CP C TP
        return spec_tok + c_tok + tp_tok

def check_brackets(tokens: List[str]):
    depth = 0
    for t in tokens:
        if t == "[": depth += 1
        elif t == "]": depth -= 1
        if depth < 0: raise ValueError("Unbalanced brackets (extra ']')")
    # if depth != 0 and "[" in tokens:
    #     raise ValueError(f"Unbalanced brackets: depth={depth}")
    

# -----------------------------
# 長さ調整付き Derivation (ロジック統合)
# -----------------------------
def build_fixed_length_derivation(
    rng: random.Random,
    sg_list: List[str], pl_list: List[str],
    verbs: List[str], dets: List[str],
    agree_match: bool, # New name for 'pos'
    num_prior: float,
    p_wh_goal: float,
    p_c_wh: float,
    epp: bool,
    enable_merge: bool,
    enable_agree: bool,
    enable_move: bool,
    max_length: int,
) -> List[str]:
    toks: List[str] = []
    
    while len(toks) < max_length:
        # 1. Lexical Selection and Base Structure
        subj_num = "sg" if rng.random() < num_prior else "pl"
        # If agree_match=True (Positive example), subj_num == obj_num
        obj_num  = subj_num if agree_match else ("pl" if subj_num == "sg" else "sg")

        subj = DPNode(det=rng.choice(dets), noun=rng.choice(sg_list if subj_num=="sg" else pl_list),
                      num=subj_num, wh_minus=(rng.random() < p_wh_goal))
        obj  = DPNode(det=rng.choice(dets), noun=rng.choice(sg_list if obj_num =="sg" else pl_list),
                      num=obj_num,  wh_minus=(rng.random() < p_wh_goal))

        V = VLex(v=rng.choice(verbs))
        VP = VPNode(verb=V, obj=obj, subj=subj)
        T = TVLex(epp=epp, u_num=None)
        TP = TPNode(T=T, VP=VP, spec=None)

        # 2. Agree/Movement
        if enable_agree:
            agree_T(T, [VP.obj, VP.subj])

        if enable_move and epp:
            epp_move(TP)

        C = CLex(wh_plus=(rng.random() < p_c_wh if enable_move else False))
        CP = CPNode(C=C, TP=TP, spec=None)

        if enable_move:
            wh_move(CP)
            
        # 3. Linearization
        # VPの線形化の語順をランダムに決定 (V-Obj-Subj vs V-Subj-Obj)
        pos_v_obj_subj = rng.random() < 0.5
        
        current_toks = lin_CP(CP, enable_merge=enable_merge, pos_v_obj_subj=pos_v_obj_subj)
        
        # Merge有効時には、lin_CPの外部で [ CP ... ] の括弧が自動的に付く

        check_brackets(current_toks)

        # 4. Concatenation
        if current_toks:
             toks.extend(current_toks)
        else:
             break # Safety break
             
    return toks[:max_length]

# -----------------------------
# Vocab & saving
# -----------------------------
def build_vocab(verbs: List[str], dets: List[str], sg_list: List[str], pl_list: List[str], allow_move: bool=True) -> Dict[str,int]:
    # Grammar tokens (Labels, features, traces, brackets)
    toks = set(["[","]", "VP", "TP", "CP", "TR[-wh]","TR[DP]", "[-wh]"])
    
    # Feature-bearing heads/nodes
    toks.update(["DP[Num:sg]","DP[Num:pl]"]) 
    t_labels = [
        "T(uNum:sg)","T(uNum:pl)","T(uNum:u)",
        "T(+EPP,uNum:sg)","T(+EPP,uNum:pl)","T(+EPP,uNum:u)","C"
    ]
    toks.update(t_labels)

    if allow_move:
        toks.update(["MOV[+wh]","C(+wh)"])
    
    # Lexical tokens
    if STRIP_LEXICAL:
        toks.update(["V","D","N"])
    else:
        for v in verbs:
            toks.add(f"V({v})")
        for d in dets:
            toks.add(f"D({d})")
        for n in (sg_list + pl_list):
            toks.add(f"N({n})")
            
    vocab = {tok:i for i, tok in enumerate(sorted(toks))}
    return vocab

def encode(tokens: List[str], vocab: Dict[str,int]) -> List[int]:
    return [vocab[t] for t in tokens if t in vocab]

def save_vocab(vocab: Dict[str,int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for tok, tid in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{tok}\t{tid}\n")
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

# -----------------------------
# CLI
# -----------------------------
def generate_mg_sentences(
    file_dir: str,
    n: int = 100000,
    agree_match_ratio: float = 1.0, # Renamed from pos_ratio
    num_prior: float = 0.5,
    seed: int = 0,
    lex_dir: str | None = None,
    also_write_text: bool = True,
    allow_move: bool = True,
    p_wh_goal: float = 0.5,
    p_c_wh: float = 0.5,
    t_epp: bool = True,
    enable_merge: bool = True,
    enable_agree: bool = True,
    enable_move: bool = True,
    strip_lexical: bool = False,
    max_length: int = 1024,
):
    """
    Generate MG-derivational sequences (fixed max_length).
    The logic of build_derivation is integrated here.
    """
    global STRIP_LEXICAL
    STRIP_LEXICAL = strip_lexical

    rng = random.Random(seed); np.random.seed(seed)
    os.makedirs(file_dir, exist_ok=True)

    if lex_dir:
        sg_list, pl_list, verbs, dets = load_lexicon(
            lex_dir,
            fallback_sg=DEFAULT_SG_NOUNS,
            fallback_pl=DEFAULT_PL_NOUNS,
            fallback_verbs=DEFAULT_VERBS,
            fallback_dets=DEFAULT_DETS,
        )
    else:
        sg_list, pl_list, verbs, dets = (
            DEFAULT_SG_NOUNS, DEFAULT_PL_NOUNS, DEFAULT_VERBS, DEFAULT_DETS
        )

    vocab = build_vocab(verbs, dets, sg_list, pl_list, allow_move=allow_move)
    save_vocab(vocab, file_dir)

    out_ids = os.path.join(file_dir, f"mg_ids_{n}_{max_length}.txt")
    out_txt = os.path.join(file_dir, f"mg_sentences_txt_{n}_{max_length}.txt")

    with open(out_ids, "w", encoding="utf-8") as fid, \
         open(out_txt, "w", encoding="utf-8") if also_write_text else open(os.devnull, "w") as ftxt:

        for i in trange(n, desc=f"MG derivations (len={max_length}, merge={enable_merge})"):
            # agree_match=True: Subject Num == Object Num (Positive example for a number-agreement task)
            agree_match = (rng.random() < agree_match_ratio) 
            
            toks = build_fixed_length_derivation(
                rng,
                sg_list, pl_list, verbs, dets,
                agree_match=agree_match, # New variable name
                num_prior=num_prior,
                p_wh_goal=p_wh_goal,
                p_c_wh=p_c_wh if allow_move else 0.0,
                epp=t_epp,
                enable_merge=enable_merge,
                enable_agree=enable_agree,
                enable_move=enable_move,
                max_length=max_length,
            )
            ids = encode(toks, vocab)
            fid.write(" ".join(map(str, ids)) + "\n")
            if also_write_text:
                ftxt.write(" ".join(toks) + "\n")

    print(f"Wrote {out_ids}{' and ' + out_txt if also_write_text else ''}. Vocab in {file_dir}")

if __name__ == "__main__":
    fire.Fire({"generate_mg_sentences": generate_mg_sentences})