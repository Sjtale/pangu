"""Profile-aware Pangu model construction for submission-local variants.

The competition package only ships ``pangu_weather/`` while OneScience remains
the platform-provided dependency. Keep architecture adaptations that are needed
by profile checkpoints here instead of relying on local OneScience edits.
"""

from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery


def _as_int_list(value):
    return [int(v) for v in value]


def build_pangu_model(img_size, patch_size, embed_dim, num_heads, window_size):
    """Create a Pangu model and patch submission-local profile differences.

    The upstream OneScience Pangu implementation hardcodes patch recovery for
    the original ``[2, 4, 4]`` patch size. PGW-Lite uses ``[2, 8, 8]``, so we
    replace only the recovery heads inside the pangu_weather submission code.
    State-dict key names remain compatible because the replacement uses the
    same ``OneRecovery`` wrapper attributes.
    """

    patch_size = _as_int_list(patch_size)
    img_size = _as_int_list(img_size)
    model = Pangu(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=int(embed_dim),
        num_heads=_as_int_list(num_heads),
        window_size=_as_int_list(window_size),
    )

    if patch_size != [2, 4, 4]:
        model.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=tuple(img_size),
            patch_size=tuple(patch_size[1:]),
            in_chans=int(embed_dim) * 2,
            out_chans=4,
        )
        model.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, *tuple(img_size)),
            patch_size=tuple(patch_size),
            in_chans=int(embed_dim) * 2,
            out_chans=5,
        )

    return model
