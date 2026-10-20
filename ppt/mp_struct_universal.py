"""MP-STRUCT Universal: Formal language generator with featural parameterization.

Extends MP-STRUCT by parameterizing the featural composition of functional
heads (C, T, v, Top) across typological profiles, following the
Borer-Chomsky Conjecture (Borer 1984; Chomsky 2001): cross-linguistic
variation reduces to properties of functional categories.

Universal structural operations (fixed across all derivations):
  - Merge: hierarchical phrase structure (TopP > CP > TP > vP > VP > DP)
  - Binary branching (Specifier-Head-Complement via X-bar)
  - Argument structure: external argument in Spec-vP, internal in VP
  - EPP: obligatory Internal Merge to Spec-TP
  - Agree: phi-feature checking between T and subject DP

Variable featural properties of functional heads (per sentence):
  - v/VP complement linearization: head-initial vs head-final
  - C [+wh] feature strength: strong (overt movement) vs weak (in-situ)
  - C V-attraction feature: active (V2) vs inactive
  - T phi-feature set: full (Num+Per) / partial (Num) / null
  - Top EPP feature: active (topic prominence) vs inactive

Versions:
  v1: Base (profile_prob=0.5, simple structure)
  v2: profile_prob=1.0 + relative clauses + deeper embedding
  v4: v2 + language-specific operations (Dutch verb clusters,
      Chinese classifiers/aspect/topic chains)
  v5: Landmark-enhanced (strip lexical items, fix transitive frame,
      maximize functional landmark clarity). Based on Mita+2026 finding
      that dependency identification ambiguity reduction is the key driver.

Usage:
  python -m ppt.mp_struct_universal --seed 42
  python -m ppt.mp_struct_universal --profile_prob 1.0 --enable_rc
  python -m ppt.mp_struct_universal --profile_prob 1.0 --enable_rc --enable_lang_specific
  python -m ppt.mp_struct_universal --strip_lexical
"""

import random
import os
from tqdm import trange

try:
    import fire
except ImportError:
    fire = None


# ============================================================
# Token Vocabulary (Integer IDs)
# ============================================================

# --- Phrase boundary brackets (open/close pairs) ---
TopP_O, TopP_C = 0, 1
CP_O, CP_C = 2, 3
TP_O, TP_C = 4, 5
vP_O, vP_C = 6, 7
VP_O, VP_C = 8, 9
DP_O, DP_C = 10, 11

# --- Functional head categories ---
H_TOP = 12
H_C = 13
H_T = 14
H_v = 15
H_V = 16
H_D = 17

# --- Syntactic operations ---
OP_AGREE = 18
OP_TRACE = 19

# --- Movement type markers ---
MV_EPP = 20
MV_WH = 21
MV_V2 = 22
MV_TOPIC = 23
MV_IN_SITU = 24

# --- Phi-features: Number ---
F_SG = 25
F_PL = 26

# --- Phi-features: Person ---
F_1 = 27
F_2 = 28
F_3 = 29

# --- Other features ---
F_WH = 30

# --- Grammatical role markers ---
R_SUBJ = 31
R_OBJ = 32
R_IOBJ = 33

# --- Clause type ---
CL_DECL = 34
CL_INTERROG = 35

# --- Embedding marker ---
EMBED = 36

# --- Abstract lexical items (content word placeholders) ---
_LEX_START = 37
_N_LEX = 20
# Tokens 37-56: abstract lexical items

# --- v2+: Structural complexity tokens ---
REL = 57            # Relative clause marker

# --- v4+: Language-specific tokens ---
VCLUST = 58         # Verb cluster marker (Dutch)
CL_CLASSIFIER = 59  # Classifier (Chinese)
ASP_PERF = 60       # Perfective aspect (Chinese: 了-like)
ASP_PROG = 61       # Progressive aspect (Chinese: 在-like)
ASP_EXP = 62        # Experiential aspect (Chinese: 过-like)
TOPIC_CHAIN = 63    # Topic chain continuation (Chinese)

VOCAB_SIZE_V1 = 57
VOCAB_SIZE_V2 = 58
VOCAB_SIZE_V4 = 64
VOCAB_SIZE_V5 = 37  # No abstract lexical items


# ============================================================
# Helpers
# ============================================================

def _lex(rng):
    return rng.randint(_LEX_START, _LEX_START + _N_LEX - 1)


def _number(rng):
    return rng.choice([F_SG, F_PL])


def _person(rng):
    return rng.choice([F_1, F_2, F_3])


# ============================================================
# Typological Parameter Sampling
# ============================================================

PROFILES = {
    'english': {
        'head_dir': {'CP': 'init', 'TP': 'init', 'vP': 'init', 'VP': 'init'},
        'wh_movement': True,
        'v2': False,
        'agreement': 'limited',
        'topic_prominent': False,
        'lang': 'english',
    },
    'dutch': {
        'head_dir': {'CP': 'init', 'TP': 'init', 'vP': 'final', 'VP': 'final'},
        'wh_movement': True,
        'v2': True,
        'agreement': 'rich',
        'topic_prominent': False,
        'lang': 'dutch',
    },
    'chinese': {
        'head_dir': {'CP': 'init', 'TP': 'init', 'vP': 'init', 'VP': 'init'},
        'wh_movement': False,
        'v2': False,
        'agreement': 'none',
        'topic_prominent': True,
        'lang': 'chinese',
    },
}


def sample_parameters(rng, profile_prob=0.5, lang=None):
    """Sample typological parameters for one sentence.

    Args:
        rng: Random number generator.
        profile_prob: Probability of sampling from a known profile.
            When lang is also set, controls the ratio of lang-specific
            vs random parameters (1.0 = 100% lang, 0.7 = 70% lang).
        lang: If set ('english', 'dutch', 'chinese'), bias toward that
            profile. With profile_prob=1.0 (default for --lang), always
            uses the specified profile. With profile_prob<1.0, mixes
            the lang profile with random parameter sampling.
    """
    if lang is not None:
        if profile_prob >= 1.0 or rng.random() < profile_prob:
            profile = PROFILES[lang]
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in profile.items()}

    if lang is None and rng.random() < profile_prob:
        profile = rng.choice(list(PROFILES.values()))
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in profile.items()}

    params = {}
    params['head_dir'] = {'CP': 'init', 'TP': 'init'}
    vp_dir = rng.choice(['init', 'final'])
    params['head_dir']['vP'] = vp_dir
    params['head_dir']['VP'] = vp_dir
    params['wh_movement'] = rng.choice([True, False])
    params['v2'] = rng.choices([True, False], weights=[0.25, 0.75])[0]
    params['agreement'] = rng.choice(['none', 'limited', 'rich'])
    params['topic_prominent'] = rng.choices([True, False], weights=[0.3, 0.7])[0]
    params['lang'] = 'mixed'
    return params


# ============================================================
# Phrase Builders
# ============================================================

def _make_dp(rng, role, agreement, is_wh=False,
             depth=0, enable_rc=False, enable_lang=False, params=None,
             profile_prob=1.0, strip_lexical=False):
    """Build a DP with phi-features, optional wh-marking,
    optional classifier (Chinese), and optional relative clause."""
    tokens = [DP_O, H_D, role]
    feats = {}

    # Chinese classifier (v4): D + CL + N
    lang = params.get('lang', 'mixed') if params else 'mixed'
    if enable_lang and lang == 'chinese' and rng.random() < 0.4:
        tokens.append(CL_CLASSIFIER)

    if not strip_lexical:
        tokens.append(_lex(rng))

    num = _number(rng)
    feats['num'] = num

    if agreement == 'rich':
        per = _person(rng)
        feats['per'] = per
        tokens.extend([num, per])
    elif agreement == 'limited':
        feats['per'] = F_3
        tokens.append(num)

    if is_wh:
        tokens.append(F_WH)
        feats['wh'] = True

    # Relative clause (v2+): DP-internal CP modifier
    if enable_rc and depth < 2 and not is_wh and rng.random() < 0.20:
        rc_lang = params.get('lang') if params and params.get('lang', 'mixed') in PROFILES else None
        rc_params = sample_parameters(rng, profile_prob, lang=rc_lang)
        rc_params['v2'] = False
        tokens.append(REL)
        rc = generate_clause(rng, rc_params, embedded=True, depth=depth + 1,
                             enable_rc=enable_rc, enable_lang=enable_lang,
                             profile_prob=profile_prob, strip_lexical=strip_lexical)
        tokens.extend(rc)

    tokens.append(DP_C)
    return tokens, feats


def _order(head_dir, head_tokens, complement_tokens):
    """Linearize head and complement according to directionality."""
    if head_dir == 'init':
        return head_tokens + complement_tokens
    return complement_tokens + head_tokens


# ============================================================
# Clause Generation
# ============================================================

def generate_clause(rng, params, embedded=False, depth=0,
                    enable_rc=False, enable_lang=False, profile_prob=1.0,
                    strip_lexical=False):
    """Generate a single clause as a token sequence.

    The derivation proceeds bottom-up: VP -> vP -> TP -> CP -> (TopP).
    """
    hd = params['head_dir']
    wh_mv = params['wh_movement']
    v2 = params['v2'] and not embedded
    agr = params['agreement']
    topic = params['topic_prominent'] and not embedded
    lang = params.get('lang', 'mixed')

    # --- Clause type ---
    is_interrog = rng.random() < (0.35 if wh_mv else 0.20)

    # --- Transitivity ---
    if strip_lexical:
        trans = 'trans'
    else:
        trans = rng.choices(
            ['intrans', 'trans', 'ditrans'], weights=[0.25, 0.50, 0.25]
        )[0]

    # --- Wh-element placement ---
    wh_subj = is_interrog and rng.random() < 0.3
    wh_obj = is_interrog and not wh_subj and trans != 'intrans'

    # --- Generate arguments ---
    dp_kwargs = dict(depth=depth, enable_rc=enable_rc,
                     enable_lang=enable_lang, params=params,
                     profile_prob=profile_prob, strip_lexical=strip_lexical)
    subj_tok, subj_f = _make_dp(rng, R_SUBJ, agr, is_wh=wh_subj, **dp_kwargs)

    obj_tok, iobj_tok = None, None
    if trans in ('trans', 'ditrans'):
        obj_tok, _ = _make_dp(rng, R_OBJ, agr, is_wh=wh_obj, **dp_kwargs)
    if trans == 'ditrans':
        iobj_tok, _ = _make_dp(rng, R_IOBJ, agr, **dp_kwargs)

    verb_lex = None if strip_lexical else _lex(rng)

    # ---- VP: V + internal arguments ----
    v_head = [H_V] if strip_lexical else [H_V, verb_lex]

    # Chinese aspect marker (v4): post-verbal aspect
    if enable_lang and lang == 'chinese' and rng.random() < 0.4:
        asp = rng.choice([ASP_PERF, ASP_PROG, ASP_EXP])
        v_head.append(asp)

    # Dutch verb cluster (v4): auxiliary/modal + main verb in clause-final
    has_vclust = False
    if enable_lang and lang == 'dutch' and hd['VP'] == 'final' and rng.random() < 0.35:
        has_vclust = True
        aux_lex = None if strip_lexical else _lex(rng)

    obj_part = []
    if obj_tok is not None:
        if wh_obj and wh_mv:
            obj_part = [OP_TRACE]
        else:
            obj_part = list(obj_tok)
            if wh_obj and not wh_mv:
                obj_part.append(MV_IN_SITU)

    iobj_part = list(iobj_tok) if iobj_tok else []

    if hd['VP'] == 'init':
        vp_inner = v_head + obj_part + iobj_part
    else:
        if has_vclust:
            # Dutch verb cluster: OBJ V_main VCLUST V_aux (IPP-like)
            aux_head = [VCLUST, H_V] if strip_lexical else [VCLUST, H_V, aux_lex]
            vp_inner = iobj_part + obj_part + v_head + aux_head
        else:
            vp_inner = iobj_part + obj_part + v_head

    vp = [VP_O] + vp_inner + [VP_C]

    # ---- vP: external argument (trace after EPP) + v + VP ----
    v_head_tok = [H_v]
    spec_vp = [OP_TRACE]
    vp_inner2 = spec_vp + _order(hd['vP'], v_head_tok, vp)
    vp_phrase = [vP_O] + vp_inner2 + [vP_C]

    # ---- TP: moved subject + T (with Agree) + vP ----
    t_head = [H_T]
    if agr != 'none':
        t_head.append(OP_AGREE)
        t_head.append(subj_f['num'])
        if agr == 'rich':
            t_head.append(subj_f.get('per', F_3))

    epp_subj = subj_tok + [MV_EPP]
    tp_inner = epp_subj + _order(hd['TP'], t_head, vp_phrase)
    tp = [TP_O] + tp_inner + [TP_C]

    # ---- CP: (wh-phrase) + C + TP ----
    cl_marker = CL_INTERROG if is_interrog else CL_DECL
    c_head = [H_C, cl_marker]

    if v2:
        if strip_lexical:
            c_head.extend([MV_V2, H_V])
        else:
            c_head.extend([MV_V2, H_V, verb_lex])

    spec_cp = []
    if is_interrog and wh_mv:
        if wh_subj:
            spec_cp = list(subj_tok) + [MV_WH]
        elif wh_obj and obj_tok is not None:
            spec_cp = list(obj_tok) + [MV_WH]

    cp_inner = spec_cp + _order(hd['CP'], c_head, tp)
    cp = [CP_O] + cp_inner + [CP_C]

    # ---- Optional TopP ----
    if topic and rng.random() < 0.5:
        # Chinese topic chains (v4): multiple stacked TopPs
        if enable_lang and lang == 'chinese':
            n_topics = rng.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]
        else:
            n_topics = 1

        result = cp
        for i in range(n_topics):
            topic_dp, _ = _make_dp(rng, R_OBJ, agr, **dp_kwargs)
            chain_marker = [TOPIC_CHAIN] if (enable_lang and i > 0) else []
            top_inner = topic_dp + [MV_TOPIC] + chain_marker + [H_TOP] + result
            result = [TopP_O] + top_inner + [TopP_C]
        return result

    return cp


# ============================================================
# Sentence Generation (multi-clause)
# ============================================================

def generate_sentence(rng, params, max_length, profile_prob,
                      enable_rc=False, enable_lang=False,
                      strip_lexical=False, lang=None):
    """Generate a sequence of one or more clauses."""
    clause_kwargs = dict(enable_rc=enable_rc, enable_lang=enable_lang,
                         profile_prob=profile_prob, strip_lexical=strip_lexical)

    tokens = generate_clause(rng, params, **clause_kwargs)

    # Embedded clause (CP complement)
    emb_prob = 0.45 if enable_rc else 0.35
    if len(tokens) < max_length // 2 and rng.random() < emb_prob:
        emb_params = sample_parameters(rng, profile_prob, lang=lang)
        emb_params['v2'] = False
        tokens.append(EMBED)
        tokens.extend(generate_clause(rng, emb_params, embedded=True, **clause_kwargs))

    if strip_lexical:
        fill_limit = int(max_length * 0.9)
        while len(tokens) < fill_limit:
            next_params = sample_parameters(rng, profile_prob, lang=lang)
            if rng.random() < 0.35:
                tokens.append(EMBED)
                next_params['v2'] = False
                tokens.extend(generate_clause(rng, next_params, embedded=True, **clause_kwargs))
            else:
                tokens.extend(generate_clause(rng, next_params, **clause_kwargs))
    else:
        cont_prob = 0.40 if enable_rc else 0.30
        while len(tokens) < max_length // 2 and rng.random() < cont_prob:
            next_params = sample_parameters(rng, profile_prob, lang=lang)
            tokens.extend(generate_clause(rng, next_params, **clause_kwargs))

    return tokens[:max_length]


# ============================================================
# Dataset Generation
# ============================================================

def generate(
    num_sentences=100000,
    max_length=1024,
    output_dir='data/ppt/mp_struct_universal',
    output_file='universal_ids.txt',
    profile_prob=0.5,
    enable_rc=False,
    enable_lang_specific=False,
    strip_lexical=False,
    lang=None,
    seed=42,
):
    """Generate a dataset of typologically-parameterized formal language.

    Args:
        num_sentences: Number of sequences to generate.
        max_length: Maximum token count per sequence.
        output_dir: Output directory.
        output_file: Output filename.
        profile_prob: Probability of sampling from a known typological
            profile (English/Dutch/Chinese) vs. independent parameter
            sampling. 0.0 = fully random, 1.0 = only target profiles.
        enable_rc: Enable relative clauses and deeper embedding (v2+).
        enable_lang_specific: Enable language-specific operations:
            Dutch verb clusters, Chinese classifiers/aspect/topic chains (v4).
        strip_lexical: Remove abstract lexical items and fix transitive
            frame (v5). Maximizes functional landmark clarity for
            dependency identification.
        lang: Fix all sentences to a single language profile
            ('english', 'dutch', 'chinese'). Overrides profile_prob.
            For monolingual L1 experiments.
        seed: Random seed.
    """
    if lang is not None and lang not in PROFILES:
        raise ValueError(f"Unknown lang '{lang}'. Choose from: {list(PROFILES.keys())}")
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_file)

    if strip_lexical:
        vocab_size = VOCAB_SIZE_V5
        version = "v5"
    elif enable_lang_specific:
        vocab_size = VOCAB_SIZE_V4
        version = "v4"
    elif enable_rc:
        vocab_size = VOCAB_SIZE_V2
        version = "v2"
    else:
        vocab_size = VOCAB_SIZE_V1
        version = "v1"

    counts = {
        'head_init_vp': 0, 'head_final_vp': 0,
        'wh_move': 0, 'wh_insitu': 0,
        'v2': 0,
        'agr_none': 0, 'agr_limited': 0, 'agr_rich': 0,
        'topic': 0,
        'lang_english': 0, 'lang_dutch': 0, 'lang_chinese': 0, 'lang_mixed': 0,
    }
    total_tokens = 0

    lang_label = f", lang={lang}" if lang else ""
    with open(path, 'w') as f:
        for _ in trange(num_sentences, desc=f"Generating MP-STRUCT Universal ({version}{lang_label})"):
            params = sample_parameters(rng, profile_prob, lang=lang)
            tokens = generate_sentence(rng, params, max_length, profile_prob,
                                       enable_rc=enable_rc,
                                       enable_lang=enable_lang_specific,
                                       strip_lexical=strip_lexical,
                                       lang=lang)

            f.write(' '.join(str(t) for t in tokens) + '\n')
            total_tokens += len(tokens)

            if params['head_dir']['VP'] == 'init':
                counts['head_init_vp'] += 1
            else:
                counts['head_final_vp'] += 1
            if params['wh_movement']:
                counts['wh_move'] += 1
            else:
                counts['wh_insitu'] += 1
            if params['v2']:
                counts['v2'] += 1
            counts[f"agr_{params['agreement']}"] += 1
            if params['topic_prominent']:
                counts['topic'] += 1
            param_lang = params.get('lang', 'mixed')
            counts[f"lang_{param_lang}"] += 1

    n = num_sentences
    print(f"\n{'='*55}")
    print(f"  MP-STRUCT Universal ({version}) — Generation Complete")
    print(f"{'='*55}")
    print(f"  Output:          {path}")
    print(f"  Version:         {version}")
    print(f"  Features:        RC={enable_rc}, LangSpecific={enable_lang_specific}, StripLex={strip_lexical}")
    print(f"  Sentences:       {n:,}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Avg length:      {total_tokens / n:.1f} tokens/sentence")
    print(f"  Vocab size:      {vocab_size}")
    print(f"\n  Parameter Distribution:")
    print(f"    VP head-initial:  {counts['head_init_vp']:>6,}  ({100*counts['head_init_vp']/n:5.1f}%)")
    print(f"    VP head-final:    {counts['head_final_vp']:>6,}  ({100*counts['head_final_vp']/n:5.1f}%)")
    print(f"    Wh-movement:      {counts['wh_move']:>6,}  ({100*counts['wh_move']/n:5.1f}%)")
    print(f"    Wh-in-situ:       {counts['wh_insitu']:>6,}  ({100*counts['wh_insitu']/n:5.1f}%)")
    print(f"    V2:               {counts['v2']:>6,}  ({100*counts['v2']/n:5.1f}%)")
    print(f"    Agree none:       {counts['agr_none']:>6,}  ({100*counts['agr_none']/n:5.1f}%)")
    print(f"    Agree limited:    {counts['agr_limited']:>6,}  ({100*counts['agr_limited']/n:5.1f}%)")
    print(f"    Agree rich:       {counts['agr_rich']:>6,}  ({100*counts['agr_rich']/n:5.1f}%)")
    print(f"    Topic-prominent:  {counts['topic']:>6,}  ({100*counts['topic']/n:5.1f}%)")
    print(f"\n  Language Profile Distribution:")
    print(f"    English:  {counts['lang_english']:>6,}  ({100*counts['lang_english']/n:5.1f}%)")
    print(f"    Dutch:    {counts['lang_dutch']:>6,}  ({100*counts['lang_dutch']/n:5.1f}%)")
    print(f"    Chinese:  {counts['lang_chinese']:>6,}  ({100*counts['lang_chinese']/n:5.1f}%)")
    print(f"    Mixed:    {counts['lang_mixed']:>6,}  ({100*counts['lang_mixed']/n:5.1f}%)")


if __name__ == '__main__':
    if fire:
        fire.Fire(generate)
    else:
        generate()
