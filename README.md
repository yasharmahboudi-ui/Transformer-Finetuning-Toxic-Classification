# Multi-label Classification of Toxic Comments
### Detecting Co-occurring Hostility Types in User-Generated Text Using Transformer Models

---

**Authors:** Payam Latifi & Yashar Mahboudi
**Course:** Technologies for Multimodal Data Representation and Archives (Winter 2026)
**Institution:** Università degli Studi di Torino (UNITO)

---

## 📌 Abstract & Final Finding
This project designs and incrementally optimizes a transformer-based pipeline for multi-label classification of toxic comments, aimed at detecting co-occurring hostility types in a single piece of user-generated text. Using Wikipedia talk-page comments, the system learns to predict multiple toxicity labels (e.g., `toxic`, `obscene`, `insult`, `identity_hate`) under strong real-world class imbalance. 

Through a 10-step experimental roadmap, we isolate architectural and data-level improvements. Our final finding is that optimal multi-label toxic comment classification requires a holistic pipeline combining architectural improvements (RoBERTa frameworks and optimized sequence lengths) and robust data handling (dataset rebalancing and class weighting) to achieve high, balanced performance across all categories—especially rare minority classes like identity hate.

---

## 🗺️ Experimental Roadmap & Evolution
We implemented a rigorous, stepwise experimental design to track how data preprocessing and hyperparameter optimization affected our multi-label metrics:

*   **Step 1: Baseline BERT Uncased** | Initial baseline using basic configurations, 2 epochs, and a learning rate of $1\times10^{-6}$. Resulted in severe underfitting for minority classes (Macro F1: 0.36, Micro F1: 0.75).
*   **Step 2: Higher Learning Rate** | Increased learning rate to $4\times10^{-5}$ and added TensorBoard tracking. Enabled the model to begin capturing rare labels (Macro F1 surged to 0.66).
*   **Step 3: More Epochs (5 Epochs)** | Extended training to 5 epochs. Caused mild overfitting as validation loss began ascending while training loss continued downward.
*   **Step 4: Dataset Version 2 Migration** | Shifted to a streamlined 4-class architecture (`toxic`, `obscene`, `insult`, `identity_hate`), dropping low-frequency columns (`severe_toxic`, `threat`) and removing 5,666 "toxic-only" majority rows to balance the dataset. Macro F1 reached 0.78; Micro F1 hit 0.85.
*   **Step 5: Incorporated Class Weighting** | Implemented label-specific positive class weights computed by the ratio of negative-to-positive samples (Identity Hate received a heavy 6.51 weight). Surged identity hate recall to 0.83.
*   **Step 6: Increased Sequence Length (256 Tokens)** | Analyzed word count distribution (mean: 49 words, max: 1403 words). Raised max length to 256 tokens to stabilize longer textual representations without wasting GPU padding.
*   **Step 7: Implemented Early Stopping** | Programmed an evaluation loss monitor with a patience of 3 and a delta threshold of 0.01. Stabilized training, automatically halting at epoch 5.
*   **Step 8: Switched to BERT-Cased** | Preserved case boundaries to capture capitalization dynamics (e.g., shouting/aggression). Identity hate recall rose to 0.89 ($F1=0.63$), and Macro F1 hit 0.76.
*   **Step 9: RoBERTa-Base Architecture** | Replaced BERT with RoBERTa-base to exploit dynamic masking and richer representations. Identity hate recall rose to 0.93 ($F1=0.66$), achieving a Macro F1 of 0.76.
*   **Step 10: RoBERTa-Large Layer Freezing** | Scaled to RoBERTa-Large (355M parameters). To mitigate GPU constraints, we applied transfer learning by freezing the embedding layer and the first 22 encoder layers, leaving only the last 2 layers trainable. Achieved our highest identity hate recall of 0.96 ($F1=0.63$).

---

## 📊 Core Performance Metrics Summary

| Modeling Phase | Preprocessing Focus | Target Classes | Identity Hate Recall | Macro F1 | Micro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1 (Baseline)** | Unfiltered (V1) | 6 Labels | 0.00 | 0.36 | 0.75 |
| **Step 2 (High LR)** | Unfiltered (V1) | 6 Labels | 0.44 | 0.66 | 0.85 |
| **Step 4 (Data V2)** | Simplified Set | 4 Labels | 0.55 | 0.78 | 0.85 |
| **Step 8 (BERT Cased)**| Cased Inputs | 4 Labels | 0.89 | 0.76 | 0.79 |
| **Step 9 (RoBERTa-B)** | Advanced Pretrain | 4 Labels | 0.93 | 0.76 | 0.79 |
| **Step 10 (RoBERTa-L)**| Layer-Frozen Transfer | 4 Labels | **0.96** | 0.74 | 0.76 |

---

## 🛠️ Tech Stack & Libraries
*   **Frameworks:** PyTorch, Hugging Face Transformers
*   **High-Level Interface:** `simpletransformers` (utilizing `MultiLabelClassificationModel` for sigmoid-based independent classification outputs)
*   **Visualization & Metrics:** TensorBoard (Loss tracking, LRAP curves), Scikit-Learn, Seaborn, Matplotlib

---

## 📈 Key Insights & Engineering Trade-Offs
1. **The Exclusivity Boundary:** Multi-class classification utilizes a competitive `SoftMax` activation forcing a zero-sum label scenario. For content moderation, we proved independent `Sigmoid` activations per class are required since hostility categories often naturally overlap.
2. **Data-Level vs. Algorithm-Level Adjustments:** Due to having only 76 solo-labeled `identity_hate` comments, data resampling was ineffective. Shifting to loss balancing via calculated positive weights proved significantly more effective at drawing attention to minor data boundaries.
3. **The Precision-Recall Trade-off:** Incorporating high positive class weighting successfully eliminated critical false negatives (pushing identity hate recall to 96%), but it introduced a higher rate of false positives, reducing overall precision.

---
