def preprocess_frame(frame):
    """
    Convert the RGB Atari frame into a processed grayscale 84x84 image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Convert to float and normalize [0,1]
    processed = resized.astype(np.float32) / 255.0
    return processed
