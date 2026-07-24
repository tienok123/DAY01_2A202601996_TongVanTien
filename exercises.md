# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Ở temperature = 0.0, phản hồi ổn định, chính xác và gần như giống nhau giữa các lần gọi. Ở 0.7, câu trả lời tự nhiên và đa dạng hơn. Khi tăng lên 1.2 và đặc biệt là 1.8, phản hồi sáng tạo hơn nhưng bắt đầu xuất hiện nội dung lan man hoặc kém mạch lạc. Theo quan sát, từ khoảng 1.8 chất lượng phản hồi bắt đầu giảm rõ rệt.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Đối với trợ lý soạn thảo hợp đồng pháp lý, tôi sẽ chọn temperature khoảng 0.0–0.2 để đảm bảo tính chính xác và nhất quán. Đối với trợ lý viết slogan quảng cáo, tôi sẽ chọn khoảng 0.8–1.2 để tăng tính sáng tạo và đa dạng trong cách diễn đạt.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *20.000 người dùng × 2 lần gọi = 40.000 request/ngày. Mỗi lần sinh khoảng 500 token đầu ra, tương đương khoảng 20 triệu token đầu ra mỗi ngày. Theo bảng giá, GPT-4o có chi phí khoảng 200 USD/ngày (20.000 × 0.01 USD), còn GPT-4o-mini khoảng 12 USD/ngày (20.000 × 0.0006 USD). GPT-4o phù hợp cho các tác vụ yêu cầu độ chính xác cao như phân tích tài liệu pháp lý hoặc tư vấn chuyên môn. GPT-4o-mini phù hợp cho chatbot hỗ trợ khách hàng hoặc các tác vụ hỏi đáp thông thường để tiết kiệm chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Hai phản hồi có sự khác biệt rõ rệt về giọng văn và cách trình bày. Persona "nhà thơ" sử dụng nhiều hình ảnh ví von, câu văn mềm mại và hầu như không dùng thuật ngữ kỹ thuật. Persona "kỹ sư phần mềm senior" trình bày ngắn gọn, chính xác, sử dụng thuật ngữ chuyên môn và có thể đưa thêm ví dụ hoặc mã nguồn. Điều này cho thấy system prompt có thể điều khiển phong cách, mức độ kỹ thuật, độ dài và cách diễn đạt của phản hồi.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Số token do tiktoken đếm thường khác với cách ước lượng theo số từ/0.75 khoảng 10–20% tùy nội dung tiếng Việt. Nếu chỉ dùng ước lượng thô để dự đoán chi phí API, kết quả có thể sai lệch vì tokenizer chia văn bản theo nhiều đơn vị nhỏ hơn từ, đặc biệt với dấu câu và ký tự Unicode. Vì vậy, nên dùng tiktoken để tính chi phí chính xác hơn.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Streaming mang lại lợi ích lớn nhất cho chatbot văn bản và đặc biệt là trợ lý giọng nói vì người dùng nhìn thấy hoặc nghe được phản hồi ngay khi mô hình đang sinh nội dung, giúp giảm cảm giác chờ đợi. Đối với pipeline dịch tài liệu chạy ngầm vào ban đêm, streaming gần như không cần thiết vì người dùng chỉ quan tâm đến kết quả cuối cùng chứ không theo dõi quá trình sinh phản hồi.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Exponential backoff giúp giảm số lượng request gửi đồng thời khi API quá tải, từ đó giảm nguy cơ làm hệ thống tiếp tục bị nghẽn. Nếu tất cả client đều retry với khoảng thời gian cố định, chúng sẽ tiếp tục gửi request cùng lúc và tạo ra các đợt quá tải mới. Kỹ thuật jitter bổ sung một khoảng thời gian ngẫu nhiên vào mỗi lần retry để các client không gửi request đồng loạt, giúp phân tán lưu lượng và tăng khả năng phục hồi của hệ thống.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System prompt: "Bạn là trợ giảng AI thân thiện. Luôn trả lời bằng tiếng Việt, giải thích từng bước rõ ràng, sử dụng ví dụ thực tế và giữ câu trả lời ngắn gọn." Nếu bỏ yêu cầu "trả lời bằng tiếng Việt", trợ lý có thể trả lời bằng tiếng Anh hoặc ngôn ngữ khác. Nếu bỏ yêu cầu "giải thích từng bước", trợ lý sẽ trả lời ngắn hơn và có thể bỏ qua các bước trung gian, khiến người mới học khó theo dõi.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Nếu người dùng hỏi về một dự án trong nhiều lượt hội thoại và các thông tin quan trọng xuất hiện ở đầu cuộc trò chuyện, việc chỉ giữ lại 4 lượt gần nhất có thể khiến trợ lý quên các chi tiết quan trọng và trả lời sai hoặc thiếu ngữ cảnh. Một cách khắc phục là tóm tắt các lượt hội thoại cũ thành một đoạn ngắn rồi đưa đoạn tóm tắt này vào history, hoặc tăng giới hạn lưu lịch sử một cách có chọn lọc đối với những thông tin quan trọng.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
