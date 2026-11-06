# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603559
- **Họ và tên:** Nguyễn Tuấn Dũng
- **SEED** (in ở ô đầu notebook): 968

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✓ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✓ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✓ |

## 2. Kết quả chính

```
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.986   0.809      0.889        0.788   
2. BoW + MLP (deep learning)         0.976   0.933      0.954        0.947   
3a. Encoder (MiniLM) + kNN           0.892   0.933      0.912        0.974   
3b. Encoder (MiniLM) + LogReg        0.933   0.933      0.933        0.947   
3c. Encoder (MiniLM) + MLP head      0.932   0.921      0.927        0.947   
4a. LLM nhỏ zero-shot                0.950   0.950      0.950        0.950   
4b. LLM lớn zero-shot                1.000   0.900      0.947        0.947   
4c. LLM nhỏ few-shot                 1.000   0.950      0.974        0.974   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.131         0.000  
2. BoW + MLP (deep learning)          2.391         0.000  
3a. Encoder (MiniLM) + kNN          123.608         0.000  
3b. Encoder (MiniLM) + LogReg        38.685         0.000  
3c. Encoder (MiniLM) + MLP head      36.461         0.000  
4a. LLM nhỏ zero-shot              3124.596         0.007  
4b. LLM lớn zero-shot              3369.620         0.072  
4c. LLM nhỏ few-shot               5884.720         0.018
```

## 3. Dự đoán và thực tế

**dl_relu** — F1 khi bỏ ReLU. Em đoán mạng bỏ ReLU sẽ có F1 gần logistic regression vì hai lớp Linear liên tiếp có thể gộp thành một lớp. Thực tế, mạng bỏ ReLU đạt 0,960, gần MLP có ReLU (0,954) hơn logistic regression (0,889). Em đã đánh đồng việc hai mô hình có cùng dạng tuyến tính với việc chúng sẽ học ra cùng trọng số. Hai thí nghiệm dùng cách khởi tạo và thuật toán tối ưu khác nhau, nên kết quả có thể khác.

**hard_eval** — Kết quả trên 50 ca khó. Em đoán hai mô hình BoW đạt 60–75% accuracy và prompt injection sẽ là nhóm khó nhất với LLM. Thực tế, cả hai mô hình BoW chỉ đạt 54%, còn hai cấu hình LLM đạt 98% và đúng 6/6 ca prompt injection. Em đã đánh giá thấp mức độ mà cách viết né từ khóa và giọng điệu tự nhiên làm khó BoW. Với LLM có prompt phòng thủ, lỗi thực tế lại là chặn nhầm một tin thường xin địa chỉ để gửi vé hòa nhạc.

**ml_lr** — Đường loss khi lr=50. Em đoán loss sẽ dao động mạnh kéo dài vì bước cập nhật quá lớn. Thực tế, loss chỉ vọt lên ở đầu rồi giảm nhanh, kết thúc ở 0,007. Bước đầu làm mô hình nghiêng quá mạnh về phía ham, khiến các tin spam bị phạt nặng. Sau đó, gradient được tính lại từ dự đoán mới và các bước cập nhật tiếp theo kéo loss xuống.

## 4. Phân tích 50 case khó

BoW + MLP tụt F1 nhiều nhất, từ 0,947 trên tập SUB xuống 0,410 trên bộ 50 ca khó, giảm 0,537. Nhóm yếu nhất là né từ khóa, trong đó mô hình bỏ sót cả 6/6 tin spam.

- **Ca #29** — "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…" Tin thay chữ bằng số, chẳng hạn won thành w0n và gift thành g1ft. BoW coi chúng là các token khác, nên những đặc trưng quen thuộc như won và gift không được bật.
- **Ca #31** — "U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d…" Dấu chấm làm bộ tách từ chia cash và reward thành các chữ cái rời. BoW không giữ thứ tự token, nên MLP mất tín hiệu từ cụm cash reward dù người đọc vẫn nhận ra.

## 5. Thử thách cá nhân

**Đề theo SEED:** Đề #3 — Thay đổi lớp ẩn của MLP thành 4 và 512 nơ-ron. So sánh số tham số, F1 trên tập test và thời gian huấn luyện.

**Cách làm:** Em tạo hai MLP với cùng đầu vào bag-of-words 2.000 chiều và đặt lại cùng SEED trước mỗi lần khởi tạo. Em giữ nguyên tập dữ liệu, 30 epoch, learning rate 0.001 và batch size 64. Sau khi huấn luyện, em đếm tham số và tính F1 của cả hai mạng trên cùng tập test.

**Kết quả**

| Số nơ-ron ẩn | Số tham số | F1 test | Thời gian huấn luyện |
| ---: | ---: | ---: | ---: |
| 4 | 8.009 | 0,953 | 5,70 giây |
| 512 | 1.025.025 | 0,949 | 54,37 giây |

**Giải thích:** Với \(h\) nơ-ron ẩn, lớp đầu có \(2000h+h\) tham số và lớp đầu ra có \(h+1\), nên tổng là \(2002h+1\). Trong lần chạy này, mạng 512 nơ-ron có khoảng 128 lần số tham số và mất khoảng 9,5 lần thời gian huấn luyện, trong khi F1 thấp hơn 0,004. Em chọn mạng 4 nơ-ron cho bài này vì chất lượng tương đương trong lần đo, đồng thời mạng nhỏ tốn ít bộ nhớ và huấn luyện nhanh hơn.

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + MLP**. Trong lần đo của em, mô hình có p95 = 2,391 ms/tin, dưới SLA 50 ms, và tốc độ xử lý cả lô khoảng 14.491 tin/s, cao hơn tải trung bình 116 tin/s. Mô hình đạt F1 = 0,954 trên toàn tập test, cao nhất trong các phương án đạt SLA, và không tốn phí API.

Đánh đổi em chấp nhận là chất lượng giảm trên tin khó. F1 của BoW + MLP giảm từ 0,947 trên SUB xuống 0,410 trên bộ 50 ca khó, và mô hình bỏ sót cả 6 tin spam né từ khóa. LLM nhỏ zero-shot làm tốt hơn trên bộ khó (F1 = 0,983), nhưng p95 khoảng 3,12 giây, vượt SLA. Trước khi triển khai, em sẽ đo lại p95 của BoW + MLP dưới tải thực tế, kể cả lúc cao điểm.

## 7. AI log và điều đã học

### 🤖 AI log

| Phần | Em hỏi AI gì? | Em nhận được gì? | Em đã sửa hoặc kiểm chứng gì? |
|---|---|---|---|
| 1 — Machine Learning | Bag-of-words, sigmoid, phép `@` và gradient descent hoạt động thế nào? Vì sao cần F1? | AI giải thích các công thức bằng ví dụ từ `free`. | Em đối chiếu với kết quả chạy. LogReg đạt F1 `0,889`, precision `0,986` và recall `0,809`. Với `lr=50`, em sửa nhận xét vì loss chỉ vọt lên lúc đầu rồi giảm nhanh. |
| 2 — Deep Learning | Tham số là gì? ReLU có tác dụng gì? Làm sao đọc đồ thị để nhận ra overfit? | AI giải thích trọng số, bias, ReLU, vòng lặp huấn luyện. | Em đối chiếu với đồ thị có val loss thấp nhất ở chỉ số epoch `9`. Em cập nhật nhận xét sau khi thấy mạng bỏ ReLU đạt F1 `0,960`, còn mạng có ReLU đạt `0,954`. |
| 3 — Embedding | Embedding khác BoW thế nào? Cosine và kNN hoạt động ra sao? | AI giải thích cách biểu diễn nghĩa của câu. | Em xem biểu đồ PCA và kết quả 6 tin ngụy trang. Em sửa dự đoán ban đầu vì BoW và Encoder + LogReg đều đúng `5/6`. |
| 4 — Code và cấu hình API | Viết các hàm gọi API, tạo prompt, đọc nhãn và ước tính chi phí. Thêm key cho endpoint custom thế nào? | AI viết code mẫu cho TODO 4.1–4.4, giải thích lỗi khi dùng `"spam" in response.lower()` và hướng dẫn lưu key trong Colab Secrets. | Em dùng provider `custom` và chạy được các thí nghiệm zero-shot và few-shot trên cùng 40 tin SUB. Các lượt chạy em đối chiếu đều có `invalid=0`. |
| 4 — So sánh LLM | Kết quả model lớn, model nhỏ và few-shot có khớp dự đoán không? | AI giúp so sánh F1, chi phí và giải thích vì sao phải dùng cùng tập SUB khi so với MiniLM. | Em cập nhật theo bảng cuối. Model nhỏ zero-shot đạt `0,950`, model lớn đạt `0,947`, còn model nhỏ few-shot đạt `0,974`. |
| 5 — Pareto và chọn mô hình | Đọc biểu đồ Pareto thế nào và chọn mô hình nào nếu p95 phải dưới 50 ms? | AI giải thích cách so sánh độ trễ với F1, tính chi phí LLM theo tháng và đề xuất kết hợp BoW + MLP với LLM để rà soát tin khó. | Em dùng bảng benchmark mới nhất để sửa số liệu. BoW + MLP đạt p95 `2,391 ms`, F1 test `0,954` và tốc độ cả lô khoảng `14.491 tin/s`. |
| 5 — Stress test | Mô hình nào tụt nhiều nhất? Prompt phòng thủ có giúp không? | AI giúp phân tích theo nhóm lỗi và gợi ý dùng bộ phân loại độc lập để rà soát các tin có kết quả bất đồng. | Em chạy lại 50 ca gốc và sửa kết luận. Cả hai prompt đều đúng `49/50` và đúng `6/6` ca prompt injection. Bản phòng thủ chặn nhầm tin ham `#9`. |
| 5 — Ca kiểm thử bổ sung | Gợi ý đúng 5 tin nhắn để thử làm khó mô hình tốt nhất. | AI đề xuất 6 tin mẫu và cách lọc kết quả theo ID để tìm ca sai. | Em yêu cầu sửa số lượng từ 6 xuống đúng 5. |

### 📚 Điều em hiểu ra hôm nay

Trước đây em nghĩ hai mô hình có cùng dạng toán học thì điểm F1 sẽ gần nhau, nhưng thí nghiệm bỏ ReLU giúp em thấy cách khởi tạo và tối ưu cũng ảnh hưởng đến kết quả. Em cũng hiểu vì sao một mô hình đạt F1 cao trên tập test thường vẫn có thể bỏ sót cả một nhóm tin viết né từ khóa. Khi chọn mô hình để triển khai, em cần xem nó sai ở đâu và mất bao lâu để xử lý mỗi tin, bên cạnh điểm F1 chung.
