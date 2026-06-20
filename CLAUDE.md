# CLAUDE.md — bối cảnh dự án (đọc đầu mỗi phiên)

## Dự án
Nghiên cứu & thử nghiệm **LeWorldModel (LeWM)** — world model kiểu JEPA học trực tiếp từ pixel
([arXiv 2603.19312](https://arxiv.org/html/2603.19312v1)) — áp dụng cho **xe tự lái mô phỏng 2D**.

Quyết định đã chốt (xem chi tiết ở [docs/00-brainstorm.md](docs/00-brainstorm.md)):
- Observation = **ảnh top-down (pixels)**.
- Task = **lái tới đích + né vật cản** (goal-conditioned navigation), điều khiển bằng **MPC + CEM** trong latent.
- LeWM: ViT-tiny encoder → `z`; predictor transformer `ẑ_{t+1}=pred(z_t,a_t)` (action qua AdaLN);
  loss = `||ẑ_{t+1}−z_{t+1}||² + λ·SIGReg`; ~15M params, 1 GPU.

## Cách làm việc (QUAN TRỌNG)
- Đây là dự án **học tập**. **Tác giả tự code; Claude KHÔNG viết code.**
- Claude đóng vai **người thầy/hướng dẫn**: giải thích khái niệm, thiết kế, lộ trình, và **review** code của tác giả.
- Tác giả tự nhận coding còn yếu → giải thích kiên nhẫn, chia nhỏ từng bước.
- Chỉ viết code khi tác giả **nói rõ** muốn đổi thoả thuận này.

## Lộ trình (theo dõi tiến độ ở README.md)
GĐ0 simulator → GĐ1 thu data (o,a,o') → GĐ2 train world model → GĐ3 planner CEM+MPC → GĐ4 vòng điều khiển + đánh giá.
Nguyên tắc: mỗi GĐ chạy được & kiểm chứng được rồi mới sang bước sau.

## Ghi chú vận hành
- Lưu ghi nhớ/bối cảnh quan trọng vào repo (file này + `docs/`), KHÔNG chỉ để ở local memory,
  vì tác giả làm việc trên nhiều máy.
