import matplotlib.pyplot as plt

def show_sample(image, mask, index=0):
    image_np = image.numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
    mask_np = mask

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image_np)
    ax[0].set_title(f"Image {index}")
    ax[0].axis('off')

    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title(f"Mask {index}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()