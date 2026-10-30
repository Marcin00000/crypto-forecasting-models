"""Early dataset checks that fail with an actionable message before data loading."""
import json
import os

# Datasets built from a narrow feature_set (E114 lr_optimal, E115 signal_optimal)
# only contain the columns of one standalone model family.
NARROW_FEATURE_SETS = ('signal_optimal', 'lr_optimal')


def assert_fusion_compatible_dataset(data_path, submodel_dirs):
    """Reject submodel fusion on a narrow dataset.

    LR submodels read their own feature columns out of the Signal input. On a narrow
    dataset most of those columns are absent, so they would be zero-filled and exceed
    ``submodel_max_missing_ratio`` (or, if permitted, feed the fusion head noise).
    """
    if not submodel_dirs:
        return
    meta_path = os.path.join(str(data_path), 'metadata.json')
    if not os.path.isfile(meta_path):
        return
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata.get('feature_set') in NARROW_FEATURE_SETS:
        raise ValueError(
            f"Submodel fusion needs a superset dataset (E109/E110, feature_set 'max'): "
            f"dataset '{metadata.get('data_id')}' has feature_set "
            f"'{metadata.get('feature_set')}' with {metadata.get('n_features')} columns, "
            f"so most LR submodel inputs would be zero-filled. "
            f"Datasets with feature_set in {list(NARROW_FEATURE_SETS)} are for "
            f"standalone Signal models only (omit --submodel_dirs)."
        )
