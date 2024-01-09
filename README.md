Tải pretrain model của Facenet: https://drive.google.com/file/d/1SVoE5QoolBxKwN8CrHre2f0BjioJYcb3/view?usp=sharing

0. pip3 install -r requirements.txt
1. Tiền xử lý ảnh (cắt mặt từ ảnh gốc):
python3 src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25 

2. Train model 
python3 src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000

3. Test
python3 src/face_rec_cam.py 
