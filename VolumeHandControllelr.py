import cv2
import numpy as np

# Khởi tạo mô hình KNN
knn = cv2.ml.KNearest_create()

# Giả sử X_train và y_train đã được chuẩn bị sẵn
# X_train là ma trận đặc trưng của hình ảnh, y_train là nhãn tương ứng

# Huấn luyện mô hình
knn.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.float32))

# Lưu mô hình nếu cần
knn.save('knn_model.xml')

# Để sử dụng mô hình đã huấn luyện
knn = cv2.ml.KNearest_load('knn_model.xml')
