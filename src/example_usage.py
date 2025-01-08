import numpy as np
from seu_tca import SEU_TCA

# Example usage
Xs = np.random.rand(100, 50)
Xt = np.random.rand(100, 50)
Ys = np.random.randint(0, 2, size=(100, 1))
Yt = np.random.randint(0, 2, size=(100, 1))

model = SEU_TCA(kernel_type='rbf', dim=10, gamma=0.1)
acc, y_pred = model.fit_predict(Xs, Ys, Xt, Yt)

print(f"Accuracy: {acc}")