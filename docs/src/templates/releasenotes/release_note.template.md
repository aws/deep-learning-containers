# {{ dlc_long }} for {{ title }}

{{ dlc_long }} for {{ platform_display }} are now available with {{ framework }} {{ version }}.

## Announcement
{% for item in announcement %}
- {{ item }}
{% endfor %}

## Core Packages

| Package | Version |
| ------- | ------- |
{% for key, value in packages.items() -%}
| {{ display_names.get(key, key) }} | {{ value }} |
{% endfor %}

## Security Advisory

{{ aws }} recommends that customers monitor critical security updates in the [{{ aws }} Security Bulletin]({{ security_bulletin_url }}).

## Reference

### Docker Image URIs
```
{% for uri in image_uris %}
{{ uri }}
{% endfor %}
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository]({{ github_repo_url }})
{% if known_issues %}

## Known Issues
{% for issue in known_issues %}
- {{ issue }}
{% endfor %}
{% endif %}
