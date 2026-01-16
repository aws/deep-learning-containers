# AWS Deep Learning Containers for {{ framework_name }} {{ version }} {{ job_type_title }} on {{ platform_title }}

[AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/) (DLCs) for {{ platform_title }} are now available with {{ framework_name }} {{ version }}{% if packages.cuda %} and support for CUDA {{ packages.cuda }}{% endif %} on {{ os_title }}.

## Release Notes

- Python {{ python }} support
- {{ os_title }} support
    {% if packages.cuda -%}
- CUDA {{ packages.cuda }} support
    {% endif %}

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Environment

- Python: {{ python }}
- OS: {{ os_title }}
    {% if architecture == "arm64" -%}
- Architecture: ARM64
    {% endif %}

{% if packages %}

## Packages

{% for key, value in packages.items() -%}

- {{ key }}: {{ value }}
    {% endfor %}
    {% endif %}

## Available Image Tags

{% for tag in tags -%}

- `{{ tag }}`
    {% endfor %}

## Example URL

```
763104351884.dkr.ecr.<region>.amazonaws.com/{{ repository }}:{{ tags[0] }}
```

{% for section_key, section_title in sections_config.items() %}
{% if sections.get(section_key) %}

## {{ section_title }}

{% for item in sections[section_key] -%}

- {{ item }}
    {% endfor %}
    {% endif %}
    {% endfor %}
