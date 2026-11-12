"""AMP (Aggregation-Method-Prediction) instances.

Plan §4.1 (TH §3.3.1.2 p.36-37; MLA §4.2); step A11 of plan §9.3.

  Instance : one case -> the 27 CDMCR features (TH Table 3.2).
  Labels   : (O_MR, O_HAC, O_WC, O_SP, O_DA) in {0,1}^5, in this order
             (TH p.37), O_AG = 1 iff f_AG(R) = a*.

Ablation rows (plan §5.4): the feature columns of the dropped groups and
the labels of the dropped methods are removed; O_DA is recomputed over the
methods kept (TH p.48). O_MR is always kept.

The decision protocol (choose the method with the highest probability) is
in cdm.models.multilabel. Nothing here trains a model.
"""
import numpy as np

from cdm.approaches import instances as I


def amp_xy(inst, row=1, oq41=None, config=None):
    """X and Y of AMP for one ablation row (row 1 = the full version).

    inst : cdm.approaches.instances.Instances (from `build_instances`).
    row  : 1..11 of plan §5.4, or an AblationConfig.
    oq41 : plan §8 OQ41 (features 16/17); default: the config value.

    Returns instances.XY with
      X : (n_cases, number of kept features); row 1: 27 columns.
      Y : (n_cases, number of kept labels), 0/1 ints; row 1: 5 columns in
          the order MR, HAC, WC, SP, DA.
      target_names : "O_<method>".
    """
    cfg = I.ablation_config(row)
    oq41 = I.oq41_from_config(config) if oq41 is None else oq41
    X, names = I.masked_features(inst, cfg, oq41)
    rules = I.rules_kept(cfg)
    methods = I.methods_kept(cfg)
    Y = np.column_stack([inst.outcomes(m, rules) for m in methods]).astype(int)
    return I.XY(X=X, Y=Y, feature_names=names,
                target_names=tuple("O_" + m for m in methods), config=cfg,
                case_ids=list(inst.case_ids), domains=list(inst.domains))
