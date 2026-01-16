# Framework Support Policy

[AWS Deep Learning Containers](../index.md) simplify image configuration for deep learning workloads and are optimized with the latest frameworks, hardware, drivers, libraries, and operating systems. This page details the framework support policy for DLCs.

## Supported Frameworks

| Framework                   | Version           | GA Date           | EOP Date           |
| --------------------------- | ----------------- | ----------------- | ------------------ |
| {% for row in supported -%} |                   |                   |                    |
| {{ row.framework }}         | {{ row.version }} | {{ row.ga_date }} | {{ row.eop_date }} |
| {% endfor %}                |                   |                   |                    |

## Unsupported Frameworks

| Framework                     | Version           | GA Date           | EOP Date           |
| ----------------------------- | ----------------- | ----------------- | ------------------ |
| {% for row in unsupported -%} |                   |                   |                    |
| {{ row.framework }}           | {{ row.version }} | {{ row.ga_date }} | {{ row.eop_date }} |
| {% endfor %}                  |                   |                   |                    |
