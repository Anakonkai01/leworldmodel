# Giai đoạn Brainstorm — Bản đồ tư duy

> Ghi lại phiên brainstorm đầu tiên (2026-06-20). Đọc file này trước khi bắt đầu code.

## LeWorldModel (LeWM) là gì

World model kiểu **JEPA** học **trực tiếp từ pixel** — không reconstruct ảnh, không cần reward,
không cần encoder pretrained. Ba thành phần:

- **Encoder** (ViT-tiny ~5M): ảnh quan sát `o_t` → embedding `z_t = enc(o_t)`
- **Predictor** (transformer 6 lớp ~10M): dự đoán latent kế tiếp `ẑ_{t+1} = pred(z_t, a_t)`;
  action được nhồi vào mỗi lớp qua **AdaLN** (zero-init cho ổn định)
- **Loss 2 thành phần**:
  ```
  L = ||ẑ_{t+1} − z_{t+1}||²   +   λ · SIGReg(Z)        (λ ≈ 0.1)
  ```
  - `L_pred`: MSE giữa latent dự đoán và latent thật của bước sau
  - `SIGReg`: chiếu embedding lên M hướng ngẫu nhiên + kiểm định chuẩn (Epps–Pulley),
    ép phân phối latent về **Gaussian đẳng hướng** → chống "collapse" mà KHÔNG cần EMA / stop-grad
- **Điều khiển**: **MPC + CEM** trong không gian latent
  1. Encode trạng thái đầu: `z_1 = enc(o_1)`
  2. Rollout latent bằng predictor (tưởng tượng)
  3. Tối thiểu hoá chi phí cuối: `C = ||ẑ_H − z_g||²`
  4. Thực thi vài action rồi lập kế hoạch lại (MPC)
- ~15M params, train được trên **1 GPU**.

**Trực giác cốt lõi:** world model thay thế simulator *trong đầu xe*. Khi lái thật, xe không được
"tua thử" trong simulator để chọn action — nó *tưởng tượng* hậu quả bằng predictor trong latent space.

## Vòng lặp tổng thể — 5 mảnh ghép

```
   ┌─────────────────────────────────────────────┐
   │                                             │
 [1] SIMULATOR ──render──▶ ảnh top-down o_t      │
   │  (xe, đích, vật cản)                        │
   │      ▲                                       │
   │   action a_t                                 │
   │      │                                       │
 [5] CONTROL LOOP ◀── plan ── [4] PLANNER (CEM+MPC)
   │                              │ dùng          │
   │                              ▼               │
 [2] DATA (o,a,o') ──train──▶ [3] WORLD MODEL ────┘
                               (encoder + predictor + SIGReg)
```

## Lộ trình 5 giai đoạn

Nguyên tắc: mỗi giai đoạn đều chạy được & kiểm chứng được trước khi sang bước sau.
*Không bao giờ viết hàng trăm dòng rồi mới chạy lần đầu.*

| GĐ | Xây gì | Kiểm chứng "đã đúng" |
|----|--------|----------------------|
| **0** | Simulator 2D + render ảnh | Lái xe tay, thấy xe chạy, đụng vật cản thì dừng |
| **1** | Thu thập dữ liệu: chạy ngẫu nhiên, lưu `(ảnh, action, ảnh_kế)` | Mở vài ảnh xem, đếm transition |
| **2** | Encoder + Predictor + loss (train world model) | Loss giảm; SIGReg không cho latent sập về 1 điểm |
| **3** | Planner CEM+MPC trong latent | Cho 1 đích gần, xem xe có lái tới không |
| **4** | Ráp vòng điều khiển + né vật cản + đánh giá | Tỉ lệ tới đích, số lần va chạm |

## Ba quyết định thiết kế then chốt

**(a) Động học xe — bắt đầu bằng "unicycle".**
Trạng thái `(x, y, θ)`, action `(vận tốc tới, vận tốc quay)`. Dễ debug. Nâng lên *bicycle model*
(có góc lái bánh trước) sau.

**(b) Góc nhìn camera — egocentric vs global.**
- *Global*: camera cố định nhìn cả bản đồ, xe là chấm di chuyển. Dễ làm & debug planner hơn lúc đầu.
- *Egocentric*: ảnh luôn đặt xe ở giữa, xoay theo hướng xe. LeWM dùng cách này vì giúp model học
  "luật vật lý" độc lập vị trí tuyệt đối. Tốt hơn về lâu dài.

**(c) Biểu diễn đích (goal) — chỗ tinh tế nhất.**
Planner cần `z_g`. Cách làm: **render một ảnh của trạng thái đích** (xe đứng ở vị trí đích) rồi
cho qua encoder → `z_g`. Tức "đích" cũng chỉ là một bức ảnh như mọi quan sát khác. Hiểu điểm này
là hiểu linh hồn của goal-conditioned JEPA.

## Công cụ cần
- **Python + PyTorch** (world model)
- **NumPy + pygame** (simulator: vẽ + bắt phím) hoặc `matplotlib`/`PIL` nếu không cần lái tay
- **GPU** hữu ích ở GĐ 2; GĐ 0–1 chạy CPU thoải mái

## Nguồn
- [LeWorldModel paper (arXiv)](https://arxiv.org/html/2603.19312v1)
- [LeWorldModel — Medium](https://medium.com/@adnanmasood/leworldmodel-and-the-case-for-stable-latent-world-models-0e4c33ca0f3c)
- [Awesome World Models for Autonomous Driving](https://github.com/LMD0311/Awesome-World-Model)
