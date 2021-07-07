import os

import toml


# If CODEBUILD_SRC_DIR is not set, use relative path to find the dlc_developer_config.toml file
DLC_DEVELOPER_CONFIG = os.path.join(
    os.getenv("CODEBUILD_SRC_DIR", os.path.join(os.getcwd(), '..', '..')), "dlc_developer_config.toml"
)


def parse_dlc_developer_configs(section, option, tomlfile=DLC_DEVELOPER_CONFIG):
    data = toml.load(tomlfile)

    return data.get(section, {}).get(option)
