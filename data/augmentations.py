from torchvision import transforms

def get_train_transforms(size, mean, std, config=None):
    config = config or {}
    transform_list = []

    if config.get("horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if config.get("rotation", True):
        transform_list.append(transforms.RandomRotation(degrees=10))
    if config.get("color_jitter", True):
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    if config.get("random_resized_crop", True):
        transform_list.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0)))
    if config.get("gaussian_blur", True):
        transform_list.append(transforms.GaussianBlur(kernel_size=(3, 3)))
    if config.get("random_affine", True):
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(transform_list)

def get_val_transforms(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
