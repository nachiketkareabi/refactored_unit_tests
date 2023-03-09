def get_config():
    import os
    import yaml

    config_path = os.path.join(os.getcwd(), "data_module", "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
