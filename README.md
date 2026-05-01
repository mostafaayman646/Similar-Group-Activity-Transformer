# Football Tactical Retrieval Engine (Ongoing Project)

![Status](https://img.shields.io/badge/Status-Research%20%2F%20WIP-orange?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Sports%20Analytics-1E90FF?style=flat-square)

> **Status:** Active Development / Research Phase
> 
> **Goal:** To build a state-of-the-art, self-supervised Transformer model capable of understanding continuous multi-agent football tracking data to retrieve matching tactical shapes and plays from a vector database.

## Project Overview
This project tackles the complex challenge of spatial-temporal pattern recognition in football (soccer). By processing raw tracking data from the **FIFA World Cup 2022**, the system embeds 10-second sequences of 22 players and the ball into a high-dimensional vector space. 

Instead of relying on rigid grid-based CNNs or manually labeled data, this architecture uses **Continuous Spatial Tokenization** and **Contrastive Learning** to teach a neural network how to inherently understand tactical momentum, team shape, and geometry.

---

## What We Have Achieved So Far

### 1. Data Engineering Pipeline
Handling high-frequency multi-agent tracking data is notoriously memory-intensive. We have built a highly optimized, custom ETL pipeline:
* **`dataTensor.py`**: A fast-forwarding extraction engine that syncs Event Data timestamps with Tracking Data (bz2/JSON) to extract specific plays. Normalizes 105x68m pitch coordinates and saves matches as highly compressed PyTorch tensors.
* **`dataset.py`**: A custom PyTorch `Dataset` that perfectly slices continuous floats (X,Y coordinates) and categorical integers (Player Roles), while dynamically padding/truncating sequences to exactly 100 frames.

### 2. The Temporal Baseline Model
We have designed and trained a **Temporal Baseline Encoder** (`model.py`) to prove the viability of continuous tokenization.
* **Spatial Tokenizer:** Instead of snapping players to a grid (like older models), a custom MLP embeds the raw, continuous `(X, Y)` coordinates fused with a Role Embedding (0=Home, 1=Away, 2=Ball).
* **Time Stamp Injection:** Uses mathematical Positional Encoding (sine/cosine waves) to give the Transformer an "internal stopwatch," allowing it to understand the chronological sequence of a player's run.
* **Temporal Transformer:** Processes all 23 agents independently across 100 frames using Self-Attention, capturing individual running speeds, curves, and biomechanical momentum, before pooling them into a single 128-dimensional play embedding.

### 3. Self-Supervised Training (Contrastive Learning)
Because tactical formations are highly subjective, we train the model without labels using an **InfoNCE Contrastive Loss** framework (`train.py`).
* **Custom Football Augmentation:** Dynamically generates "Positive Matches" on the fly by applying Y-axis pitch mirroring, tracking data noise (spatial jitter), and random player dropout.
* **The Matching Game:** The model is forced to push an Anchor play and its augmented Positive together in the 128d space, while repelling the 7 other random plays in the batch (Negatives) using a strict temperature scaling mechanism.

### Key Research Findings (Baseline Evaluation)
By mapping the baseline vectors and visualizing the nearest neighbors, we achieved a crucial architectural breakthrough:
* **Success:** The Temporal-only model perfectly learns **momentum and trajectory**. It successfully groups plays where players make similar overlapping runs or hooking curves.
* **Limitation:** It completely fails to understand **pitch location and team density** (e.g., retrieving a crowded set-piece to match an open-field counter-attack). 
* **Conclusion:** This visually proved that averaging individual temporal paths destroys spatial context, validating the absolute necessity of our next architectural step: **Social Attention**.

---

## Next Steps (Currently Working On)

The project is currently transitioning from the Baseline phase to the next Hybrid Architecture. 

1. **The Social Set-Transformer:** 
   Building a new attention block that completely ignores time, but allows the 23 agents to "look" at each other in a single frame. This will teach the model the concept of spacing, density, and tactical formations (e.g., a 4-4-2 block vs. a 4-3-3 press).
2. **Hybrid Spatial-Temporal Fusion:**
   Combining the Temporal Transformer (momentum) with the Social Transformer (geometry) to create the ultimate representation of a football play.
3. **The Vector Database:**
   Deploying the final embeddings into a retrieval system (like FAISS) to allow users to input a query play and instantly find the closest tactical matches from the entire World Cup dataset.

---

## Repository Structure
* `dataTensor.py` - Raw data parsing, normalization, and tensor extraction.
* `dataset.py` - PyTorch DataLoader, memory-safe batching, and tensor typing.
* `DataBase/` - Directory containing the processed `.pt` tensor files for each match.
* `Temporal_Baseline_Encoder/`
  * `model.py` - The PyTorch neural network architecture (Positional Encoding, Tokenizer, Transformer).
  * `train.py` - Contrastive learning loop, data augmentation, and model optimization.