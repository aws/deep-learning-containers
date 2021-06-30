import os

import toml


DLC_DEVELOPER_CONFIG = os.path.join(os.getenv("CODEBUILD_SRC_DIR"), "dlc_developer_config.toml")


def parse_dlc_developer_configs(section, option, tomlfile=DLC_DEVELOPER_CONFIG):
    data = toml.load(tomlfile)

    return data.get(section, {}).get(option)
