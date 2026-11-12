"""DAP (Direct-Answer-Prediction) instances.

Plan §4.2 (TH §3.3.1.3 p.38; MLA §4.3); step A11 of plan §9.3.

  Instance : one case -> the 27 CDMCR features + the answers chosen by the
             methods, (f_MR, f_HAC, f_WC, f_SP, f_DA), in this order
             (TH p.38).
  Target   : the correct answer a*.

Answer encoding (plan §4.2, §8 OQ5; binary cases only): the first answer
of the case, a1 = Case.answers[0], is coded 1 and the other answer 0. The
target and the f_AG features use the same coding. Which answer is a1 is
decided when the data is built (cdm.data.worldA: switch `answer_coding`;
OQ5 (a) natural source coding: True / home team / first-listed candidate
is a1; (c) random per case). So the OQ5 options are compared by building
the data under each coding, not here.

More than two answers: the plan defines the coding for binary data only
("Binary classification for |A|=2 (all World-A data)", plan §4.2). The
thesis says DAP is then a multi-class task (TH p.38) but a case-level
classifier needs one answer coding shared by all cases, and neither source
gives one for cases with different answer sets. `dap_xy` raises
NotImplementedError for such cases; nothing is guessed.

Ablation rows (plan §5.4): the feature columns of the dropped groups and
the f_AG columns of the dropped methods are removed; f_DA is recomputed
over the methods kept (TH p.48). f_MR is always kept.

Nothing here trains a model.
"""
import numpy as np

from cdm.approaches import instances as I


def encode_answer(answer, answers):
    """0/1 code of an answer of a binary case: 1 for answers[0] (a1), 0 for
    answers[1]. Plan §4.2 / §8 OQ5."""
    answers = tuple(answers)
    if len(answers) != 2:
        raise NotImplementedError(
            f"the DAP answer coding is defined for binary cases only "
            f"(plan §4.2, OQ5); got {len(answers)} answers")
    if answer not in answers:
        raise ValueError(f"{answer!r} is not one of the answers {answers!r}")
    return 1 if answer == answers[0] else 0


def decode_answer(code, answers):
    """Inverse of `encode_answer`."""
    answers = tuple(answers)
    if len(answers) != 2:
        raise NotImplementedError(
            "the DAP answer coding is defined for binary cases only (plan §4.2, OQ5)")
    if code not in (0, 1):
        raise ValueError(f"code must be 0 or 1, got {code!r}")
    return answers[0] if code == 1 else answers[1]


def non_binary_positions(inst):
    """Positions of the cases with a number of answers other than two."""
    return [i for i, a in enumerate(inst.answers) if len(a) != 2]


def dap_xy(inst, row=1, oq41=None, config=None):
    """X and y of DAP for one ablation row (row 1 = the full version).

    inst : cdm.approaches.instances.Instances (from `build_instances`).
    row  : 1..11 of plan §5.4, or an AblationConfig.
    oq41 : plan §8 OQ41 (features 16/17); default: the config value.

    Returns instances.XY with
      X : (n_cases, kept features + kept f_AG columns); row 1: 27 + 5 = 32
          columns; the f_AG columns are 0/1 codes (`encode_answer`), in
          the order f_MR, f_HAC, f_WC, f_SP, f_DA.
      Y : (n_cases,), the 0/1 code of a*.
      target_names : ("a*",).

    Raises NotImplementedError if any case does not have exactly two
    answers (see the module docstring); use `inst.select(...)` to pass the
    binary cases only if that is what is wanted.
    """
    bad = non_binary_positions(inst)
    if bad:
        raise NotImplementedError(
            f"{len(bad)} case(s) do not have exactly two answers (first: "
            f"{inst.case_ids[bad[0]]!r}); the DAP answer coding is defined "
            "for binary cases only (plan §4.2, OQ5)")
    cfg = I.ablation_config(row)
    oq41 = I.oq41_from_config(config) if oq41 is None else oq41
    X, names = I.masked_features(inst, cfg, oq41)
    rules = I.rules_kept(cfg)
    methods = I.methods_kept(cfg)
    cols = []
    for m in methods:
        cols.append([encode_answer(a, ans) for a, ans in
                     zip(inst.method_answers(m, rules), inst.answers)])
    f = np.array(cols, dtype=float).T.reshape(len(inst), len(methods))
    y = np.array([encode_answer(c, ans) for c, ans in zip(inst.correct, inst.answers)],
                 dtype=int)
    return I.XY(X=np.hstack([X, f]), Y=y,
                feature_names=names + tuple("f_" + m for m in methods),
                target_names=("a*",), config=cfg,
                case_ids=list(inst.case_ids), domains=list(inst.domains))
