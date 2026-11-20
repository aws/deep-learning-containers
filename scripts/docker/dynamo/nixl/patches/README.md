# NIXL LIBFABRIC plugin patches

Upstream NIXL fixes that aren't in a release yet, applied when `build_libfabric_plugin.sh` builds the LIBFABRIC plugin
for EFA images: one directory per NIXL version (`<nixl-version>/`), each patch named `<n>-<topic>-nixl-<PR>.patch` after
its upstream pull request (the build links it in the image's OSS modification notice). Drop a directory once the
backends move to a NIXL release that includes its fixes. None are needed today.
