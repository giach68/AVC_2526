import numpy as np
import cv2


def compute_disparity_ssd(left, right, max_disparity=64, block_size=7):
    """
    Compute a left-to-right disparity map using block matching with SSD.

    Parameters
    ----------
    left : np.ndarray
        Left rectified grayscale image, shape (H, W), uint8 or float.
    right : np.ndarray
        Right rectified grayscale image, shape (H, W), same size as left.
    max_disparity : int
        Maximum disparity to search (inclusive start at 0, exclusive end at max_disparity).
    block_size : int
        Odd window size for matching.

    Returns
    -------
    disparity : np.ndarray
        Disparity map, float32, shape (H, W).
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Images must be grayscale")
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same shape")
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")
    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive")

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    h, w = left.shape
    half = block_size // 2

    disparity = np.zeros((h, w), dtype=np.float32)

    # Only compute where the full block fits in both images
    for y in range(half, h - half):
        for x in range(half + max_disparity, w - half):
            left_block = left[y - half:y + half + 1, x - half:x + half + 1]

            best_d = 0
            best_ssd = np.inf

            # Search along the epipolar line (same row)
            for d in range(max_disparity):
                xr = x - d
                if xr - half < 0:
                    break

                right_block = right[y - half:y + half + 1, xr - half:xr + half + 1]

                diff = left_block - right_block
                ssd = np.sum(diff * diff)

                if ssd < best_ssd:
                    best_ssd = ssd
                    best_d = d

            disparity[y, x] = best_d

    return disparity




def main():
    # Load rectified stereo pair
    left = cv2.pyrDown(cv2.imread("aloeL.jpg", cv2.IMREAD_GRAYSCALE))
    right = cv2.pyrDown(cv2.imread("aloeR.jpg", cv2.IMREAD_GRAYSCALE))

    if left is None or right is None:
        raise FileNotFoundError("Could not load left.png and/or right.png")

    disparity = compute_disparity_ssd(
        left,
        right,
        max_disparity=32,
        block_size=7
    )

    #disp_vis = normalize_disparity_for_display(disparity)
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    cv2.imwrite("disparity_ssd.png", disp_vis)
    np.save("disparity_ssd.npy", disparity)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity SSD", disp_vis)
    print("Saved disparity_ssd.png and disparity_ssd.npy")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()