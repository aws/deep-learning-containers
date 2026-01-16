# {{ framework_name }} Release Notes

Release notes for AWS Deep Learning Containers with {{ framework_name }}.

## Supported Releases

| Version | Job Type | Accelerator | Platform | Release Notes |
| ------- | -------- | ----------- | -------- | ------------- |
{% for r in supported -%}
| {{ r.version }} | {{ r.job_type }} | {{ r.accelerator }}{% if r.arch == "arm64" %} (ARM64){% endif %} | {{ r.platform }} | [View]({{ r.filename }}) |
{% endfor %}
{% if deprecated %}

## Deprecated Releases

| Version | Job Type | Accelerator | Platform | Release Notes |
| ------- | -------- | ----------- | -------- | ------------- |
{% for r in deprecated -%}
| {{ r.version }} | {{ r.job_type }} | {{ r.accelerator }}{% if r.arch == "arm64" %} (ARM64){% endif %} | {{ r.platform }} | [View]({{ r.filename }}) |
{% endfor %}
{% endif %}
