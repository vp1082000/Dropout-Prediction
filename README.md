
# Dropout prediction

## 1. BỐI CẢNH VÀ MỤC TIÊU
__Bối cảnh__

Dataset gồm 35 cột với 4400 dòng dữ liệu, mỗi dòng dữ liệu tượng trưng cho một sinh viên với các thông tin có thể được chia làm 3 nhóm lớn:
+ Nhân Khẩu học: 9 cột/Features
+ Tình hình học tập(hai kỳ học): 16 cột/Features
+ Bối cảnh kinh tế, xã hội: 3 cột/Features

Features Target: 
- Dropout (Bỏ học): 1421 giá trị - data label '1'
- Graduate (Tốt nghiệp): 2209 giá trị - data label '0'

__Ví dụ minh họa:__ Nếu mỗi năm trường đại học đều có _2000sv_ nhập học và mọi văn bằng đều yêu cầu 4 năm học thì số sinh viên theo học tại trường mọi thời điểm sẽ _~8000sv_. Nếu áp số liệu của data thì tỉ lệ Graduate _~27%_ và tỉ lệ Dropout _~18%_ (__Mức cao__). Trong khi đó để công nhận là trường đạt chuẩn thì tỉ lệ Dropout không quá _10%_.

__Mục tiêu__
- Đánh giá mức ảnh hưởng của các yếu tố về nhân khẩu học, bối cảnh kinh tế và tình hình học tập đến _Features Target_
- Xây dựng mô hình dự đoán nhằm hỗ trợ nhà trường nhận diện sớm các sinh viên có dấu hiệu sẽ _Dropout_ để có các hình thức tư vấn hỗ trợ kịp thời (hỗ trợ về tài chính hoặc chuyển ngành nếu có thể)

## ĐÁNH GIÁ ẢNH HƯỞNG CỦA CÁC FEATURE
__Chọn lọc Feature__

Cần loại bỏ bớt nhưng Feature không có ý nghĩa liên quan tới target và chọn một trong các Features có độ tương quan cao với nhau để giúp mô hình học nhanh hơn.

Trong số các Feature về Nhân Khẩu học:
- Hệ số tương quan (Correlation) giữa hai Features 'Nationality' và 'International' là ~0.92 (rất cao) => chỉ cần sử dụng một trong hai
![corr](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/c52edf9b-0d22-48f3-b02e-dc92a3d022fd)


Trong số các Feature về tình hình học tập:
- Credited: số tín chỉ cộng thêm
- Enrolled: số môn đăng kí 
- Evaluations:  số tín đánh giá 
- Approved: số môn hoàn thành
- Grade: điểm số
- ~~Without evaluations~~: số tín không được đánh giá (không có quan hệ về mặt ý nghĩa)

Các Feature về bối cảnh kinh tế sẽ đc giữ nguyên để đánh giá mức ảnh hưởng tới Feature Target

__Mô hình cơ bản__

Mô hình dùng để dự đoán là __RandomForest – Classifier__ với các chỉ số _Default_, để đánh giá độ chính xác của mô hình thì ta sẽ áp dụng chỉ số __Recall__

__Recall = TP/(TP + FN)__

_NOTE_: Chỉ số __Recall__ tính tỉ lệ dự đoán chính xác (TP) trên tổng số giá trị Dropout (TP+FN).
![CFdefault](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/a4e9e39f-14aa-4639-8c1a-865c146380c0)

Kết quả của mô hình:
- Recall on model:  0.814
- Dự báo bỏ học đúng (TP):   337 / 414 (nhầm 77 giá trị)
- Dự báo tốt nghiệp đúng (TN):   645 / 675 (nhầm 30 giá trị)

__Mức ảnh hưởng của các yếu tố__

Áp dụng tính năng Features Importance để xếp hạng mức ảnh hưởng của các yếu tố đến khả năng dự đoán của mô hình

![Effect](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/f44becb7-db3b-44e9-8e81-4cac499ef1b4)

Đánh Giá Chung:
- Nhóm tình hình học tập: Các Feature trong nhóm có đóng góp lớn trong khả năng dự đoán đại diện qua số môn học hoàn thành (approved) và điểm (grade) của các kì học (trên 0.075), ngoài ra là các công số về tình trạng đóng học và số môn học đăng ký (trên 0.025)
- Nhóm bối cảnh kinh tế: các Feature ít có đóng góp (dưới 0.025)
- Nhóm nhân khẩu học: các Feature ít có đóng góp (dưới 0.025), ngoại trừ Feature về độ tuổi nhập học (Age at enrollment) có ảnh hưởng tới khả năng dự đoán của máy

![Age](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/1973d069-545a-43bf-a5d1-1869bbca1eb3)
Nhận Xét về Độ tuổi nhập học:
1. Từ 18-20 tuổi, sinh viên có tỉ lệ bỏ học/tốt nghiệp ~30%, xuất phát từ việc chọn sai ngành nghề
=> Nên có tự vấn trước khi nhập học và kể cả khi đã nhập học để giúp sinh viên chuyển sang ngành học khác trong trường thay vì bỏ học

2. Những người có độ tuồi nhập học > 21 tuổi có xu hướng bỏ học tăng cao (21 tuổi: ~45% và tằng dần khi vượt 50% ở tuổi 24 và duy trì trên 50% ở các độ tuổi sau đó).

=> Giả thiết là nếu những sv trên 21 tuổi đều đã đi làm và quyết định quay lại học thì sẽ có hai trưởng hợp:
- Bỏ vì lý do cá nhân (tài chính, áp lực thời gian,…): chính sách học bổng, đào tạo ngắn hạn.
- Bỏ vì kiến thức giảng dạy xa rời yêu cầu thực tế: cải thiện giáo trình để bám sát thực tế.

## TỐI ƯU MÔ HÌNH 
__Tối ưu Feature__ 

Chọn ngưỡng chỉ số ảnh hưởng 0.025 => Giảm số lượng feature sử dụng từ 29 xuống 9 gồm:
- Curricular units 1st sem and 2nd sem (evaluations/ approved/ grade)
- Tuition fees up to date
- Age at enrollment
- Course

Áp dụng phương pháp BorderlineSMOTE để oversampling các giá trị thiểu số trong tập train (tạo thêm mẫu cho giá trị Dropout để máy có thể dễ dàng nhận diện). Khi áp dụng sẽ dẫn đến mô hình học lệch, dự đoán nhầm sẽ không có ảnh hưởng bởi thực tế các sinh viên bị dự đoán nhầm đó sẵn đã có ý định tốt nghiệp.

__Tối ưu thông số (parameter)__

Áp dụng GridSearchCV vào mô hình RandomForest, lấy giá trị tối ưu (scoring): recall_score


![gridCV](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/ef9da79f-d388-4534-a6f8-249a8ca3354f)
_Thông số mô hình tối ưu_


![CFhyper](https://github.com/vp1082000/Dropout-Prediction/assets/143709845/71552bf5-3b2e-497a-9a1b-05584d2a2646)

Kết quả của mô hình:
- Recall on model:  0.814  ->  0.852
- Dự báo bỏ học đúng (TP):  tăng 337  ->  353
- Dự báo bỏ học sai (FN): giảm 77  ->  61
- Dự báo tốt nghiệp sai (FP): tăng 30 -> 63   

=> Tuy giá trị FP (thực tế là Graduate nhưng dự báo là Dropout) tăng do áp dụng oversampling nhưng không ảnh hưởng bởi thực tế các sv này dù có nhận hỗ trợ tư vấn cũng vẫn tốt nghiệp. Trong khi TP (thực tế là Dropout và dự báo cũng là Dropout) tăng lên, điều này giúp nhà trường phát hiện nhiều nhất các trường hợp dự định Dropout. 






















