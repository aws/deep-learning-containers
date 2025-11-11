import docker
import datetime

client = docker.from_env()
cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)

for img in client.images.list():
    # 'Created' is epoch seconds
    created = datetime.datetime.utcfromtimestamp(img.attrs["Created"])
    if created < cutoff:
        name = img.tags[0] if img.tags else "<none>"
        print(f"🗑️  Removing {name} (created {created.isoformat()}Z)")
        try:
            client.images.remove(img.id, force=True)
        except docker.errors.APIError as e:
            print(f"(warning: failed to remove {name}: {e.explanation})")

print("\n=== Disk usage summary ===")
print(client.df())  # structured info like `docker system df`
