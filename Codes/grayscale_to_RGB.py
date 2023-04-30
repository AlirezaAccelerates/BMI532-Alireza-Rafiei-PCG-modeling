n,m,d = np.shape(X_train)

def grayscale_to_rgb(images):
    return np.repeat(images, 3, axis=-1)

X_train2  = grayscale_to_rgb(X_train)

X_train2 = np.reshape(X_train2, (n,m,d,3))
