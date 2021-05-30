import cv2 as cv
import numpy as np


def noise_salt(image: np.array, **kwargs) -> np.array:
    """Creates salty noise with a given amount

    Args:
        image (np.array): The image will be used to detive shape and calculate the absolute amount and the max noise value

    Kwargs:
        amount (float): The relative amount of salt to add (default: 0.004)

    Returns:
        np.array: Noise image with the specified amount of salt of same type as the input image
    """
    amount = kwargs.get("amount", 0.004)
    n_salt = np.ceil(amount * image.size)

    noise = np.zeros_like(image)
    salt_value = np.iinfo(image.dtype).max

    coordinates = tuple([np.random.randint(0, i - 1, int(n_salt)) for i in image.shape])
    noise[coordinates] = salt_value
    return noise


def noise_hlines(image: np.array, **kwargs) -> np.array:
    """Creates horizontal noise with a given spacing

    Args:
        image (np.array): The image will be used to derive shape and the max noise value

    Kwargs:
        spacing (int): The spacing between lines in px (default: 32)

    Returns:
        np.array: Noise image with the specified horizontal lines of same type as the input image
    """
    spacing = kwargs.get("spacing", 32)
    max_value = np.iinfo(image.dtype).max
    noise = np.zeros_like(image)
    noise[::spacing] = max_value
    return noise


def noise_vlines(image: np.array, **kwargs) -> np.array:
    """Creates vertical noise with a given spacing

    Args:
        image (np.array): The image will be used to derive shape and the max noise value

    Kwargs:
        spacing (int): The spacing between lines in px (default: 32)

    Returns:
        np.array: Noise image with the specified vertical lines of same type as the input image
    """
    spacing = kwargs.get("spacing", 32)
    max_value = np.iinfo(image.dtype).max
    noise = np.zeros_like(image)
    noise[:, ::spacing] = max_value
    return noise


def _rotate_image(image: np.array, angle: float) -> np.array:
    """Rotates the image by a given angle

    Args:
        image (np.array): The image to rotate
        angle (float): The rotation angle in degree

    Returns:
        np.array: The rotated image
    """
    ic = tuple(np.array(image.shape[1::-1]) / 2)
    R = cv.getRotationMatrix2D(ic, angle, 1.0)
    return cv.warpAffine(image, R, image.shape[1::-1], flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)


def noise_grid(image: np.array, **kwargs) -> np.array:
    """Creates rotated grid noise with a given angle and vertical and horizontal spacing.

    Args:
        image (np.array): The image will be used to derive the max noise value

    Kwargs:
        hspacing (int): The horizontal grid spacing in px (default: 32)
        vspacing (int): The vertical grid spacing in px (default: 32)
        angle (float): The grid's rotation angle in degree (default: 0.0)

    Returns:
        np.array: The grid noise image
    """
    hspacing = kwargs.get("hspacing", 32)
    vspacing = kwargs.get("vspacing", 32)
    angle = kwargs.get("angle", 0)
    h, w = image.shape[:2]
    border = int(np.sqrt(w ** 2 + h ** 2) - h)
    border_half = border // 2
    expanded_image = cv.copyMakeBorder(image, border, 0, border, 0, cv.BORDER_CONSTANT)
    # expanded_image = image
    h_noise, v_noise = noise_hlines(expanded_image, spacing=hspacing), noise_vlines(
        expanded_image, spacing=vspacing
    )
    noise = h_noise + v_noise
    return _rotate_image(noise, angle)[
        border_half : (h + border_half), border_half : (w + border_half)
    ]


def add_noise_to_image(original_image: np.array, noise_image: np.array) -> np.array:
    """Adds a noise image to an original image

    Adds the two images and applies a threshold to the result.

    Args:
        original_image (np.array): The "original" image
        noise_image (np.array): The noise image

    Returns:
        np.array: The combination of both images as 8-Bit image
    """
    noisy_edge_image = original_image + noise_image
    noisy_edge_image = cv.threshold(noisy_edge_image, 0, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    return noisy_edge_image


if __name__ == "__main__":
    shape = (480, 640)
    image = np.zeros(shape, dtype=np.uint8)

    image_salt = noise_salt(image, amount=0.01)
    image_noise_hlines = noise_hlines(image, spacing=128)
    image_noise_vlines = noise_vlines(image, spacing=128)
    image_noise_straight_grid = noise_grid(image, hspacing=128, vspacing=128, angle=0)
    image_noise_angled_grid = noise_grid(image, hspacing=128, vspacing=128, angle=45)

    full = add_noise_to_image(image_salt, image_noise_hlines)
    full = add_noise_to_image(full, image_noise_vlines)
    full = add_noise_to_image(full, image_noise_angled_grid)

    cv.imshow("image", image)

    cv.imshow("noise_salt 0.01", image_salt)
    cv.imshow("noise_hlines 128", image_noise_hlines)
    cv.imshow("noise_vlines 128", image_noise_vlines)
    cv.imshow("noise_grid 128 0deg", image_noise_straight_grid)
    cv.imshow("noise_grid 128 45deg", image_noise_angled_grid)
    cv.imshow("full", full)
    cv.waitKey(0)
