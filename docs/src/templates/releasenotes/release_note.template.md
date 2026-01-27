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

{{ aws }} recommends that customers monitor critical security updates in the [{{ aws }} Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

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
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
{% for section_key, items in optional.items() %}

## {{ display_names.get(section_key, section_key) }}
{% for item in items %}
- {{ item }}
{% endfor %}
{% endfor %}
