"""Support policy table generation."""

from utils import render_table

COLUMNS = ["Framework", "Version", "CUDA", "GA Date", "EOP Date"]


def generate(yaml_data: dict) -> str:
    """Generate supported and unsupported framework tables."""
    policy = yaml_data.get("support_policy", {})
    sections = []

    # Supported frameworks table
    supported = policy.get("supported", {})
    if supported:
        rows = []
        for framework, versions in supported.items():
            for version, info in versions.items():
                rows.append(
                    [
                        framework.title(),
                        version,
                        info.get("cuda", ""),
                        info.get("ga", ""),
                        info.get("eop", ""),
                    ]
                )
        sections.append("## Supported Frameworks\n\n" + render_table(COLUMNS, rows))

    # Unsupported frameworks table
    unsupported = policy.get("unsupported", {})
    if unsupported:
        rows = []
        for framework, versions in unsupported.items():
            for version, info in versions.items():
                rows.append(
                    [
                        framework.title(),
                        version,
                        info.get("cuda", ""),
                        info.get("ga", ""),
                        info.get("eop", ""),
                    ]
                )
        sections.append("## Unsupported Frameworks\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
