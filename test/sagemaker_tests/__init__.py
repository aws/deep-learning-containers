def get_ecr_image_region(ecr_image):
    # Handle malformed URLs with duplicate registry
    clean_url = ecr_image.replace("/644385875248.dkr.ecr.us-east-1.amazonaws.com", "")
    try:
        ecr_registry, _ = clean_url.split("/")
        return ecr_registry.split(".")[3]  # Extract region from registry URL
    except (IndexError, ValueError):
        raise ValueError(f"Invalid ECR image URL format: {ecr_image}")
