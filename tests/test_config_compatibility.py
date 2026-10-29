from posting_model.config import load_config


def test_legacy_sigma_n_is_accepted_only_during_config_loading(tmp_path):
    path = tmp_path / "legacy.toml"
    path.write_text("sigma_n = 2.5\n", encoding="utf-8")
    assert load_config(path).sigma_kappa == 2.5
