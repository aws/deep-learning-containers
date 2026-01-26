# {{ framework_display }} Release Notes

Release notes for {{ dlc_long }} with {{ framework_display }}.
{% for version, entries in releases_by_version.items() %}

## {{ framework_display }} {{ version }}

| Platform | Type | Link |
| -------- | ---- | ---- |
{% for entry in entries -%}
| {{ entry.platform }} | {{ entry.type }} | [{{ entry.title }}]({{ entry.filename }}) |
{% endfor %}
{% endfor %}
