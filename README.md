# rc-car — LeWorldModel trên xe tự lái 2D

Nghiên cứu & thử nghiệm **LeWorldModel (LeWM)** — một world model kiểu JEPA học trực tiếp từ pixel
([arXiv 2603.19312](https://arxiv.org/html/2603.19312v1)) — áp dụng cho một **xe tự lái mô phỏng trong không gian 2D**.

## Mục tiêu
- Quan sát = **ảnh top-down (pixels)**, đúng tinh thần LeWM "learn from pixels".
- Nhiệm vụ = **lái tới đích + né vật cản** (goal-conditioned navigation), khớp với cơ chế điều khiển MPC+CEM của LeWM.

## Tài liệu
- [docs/00-brainstorm.md](docs/00-brainstorm.md) — bản đồ tư duy tổng thể, lộ trình 5 giai đoạn, các quyết định thiết kế.

## Cách làm việc
Đây là dự án học tập: tác giả tự code, Claude đóng vai người hướng dẫn (giải thích, thiết kế, review).

## Trạng thái
- [x] Giai đoạn brainstorm
- [ ] GĐ 0 — Simulator 2D + render
- [ ] GĐ 1 — Thu thập dữ liệu (o, a, o')
- [ ] GĐ 2 — World model (encoder + predictor + SIGReg)
- [ ] GĐ 3 — Planner CEM + MPC trong latent
- [ ] GĐ 4 — Vòng điều khiển + đánh giá
