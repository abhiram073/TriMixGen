# Training Engine Report: TriMixGen mT5 Pipeline

This report explains the core mechanisms of the `GenerationTrainer` and why `Seq2SeqTrainer` is specifically tailored for mT5.

## 1. Seq2SeqTrainer & mT5
The Hugging Face `Trainer` is optimized for standard encoder-only (BERT) or decoder-only (Llama) models. However, mT5 is an Encoder-Decoder model. `Seq2SeqTrainer` is a specialized subclass that automatically handles the complex duality of generating sequences while computing loss. It seamlessly manages generating predictions using `model.generate()` during evaluation, which standard `Trainer` cannot do.

## 2. Teacher Forcing & Label Shifting
In autoregressive text generation, the model predicts the $n^{th}$ word based on the previous $n-1$ words. 
*   **Without Teacher Forcing:** If the model predicts the wrong word at step 2, that error cascades, ruining the context for step 3, making training impossible.
*   **Teacher Forcing:** During training, we don't feed the model its own predictions. We feed it the *ground truth* (the gold label target). This keeps the model on track.
*   **Label Shifting:** To predict the $n^{th}$ word, the decoder receives tokens up to $n-1$. The `Seq2SeqTrainer` handles this internally by automatically shifting the `labels` tensor one position to the right to create the `decoder_input_ids`.

## 3. Cross-Entropy Loss for Sequences
The model outputs a probability distribution over the entire 250,112-token vocabulary for *every single position* in the target sequence. We use standard Cross-Entropy Loss to measure the divergence between this predicted distribution and the actual target token. 
*Note: In `tokenizer.py`, we replaced padding tokens with `-100`. The PyTorch CrossEntropyLoss function is hardcoded to ignore the index `-100`, ensuring the model is not penalized for failing to predict empty space.*

## 4. CPU Optimization Strategies

### Gradient Accumulation
Training mT5 requires massive memory to store computational graphs. We cannot fit a batch size of 32 in RAM. We set `batch_size=2` to fit in memory, but doing a backward pass on only 2 samples makes the gradient highly erratic (noise). **Gradient Accumulation** (`steps=16`) means we run 16 micro-batches of size 2, adding up their gradients in memory, and only perform an optimizer step once we have a clean, stable macro-gradient of 32 samples.

### Warmup Scheduling
Because we randomly initialize the LoRA matrices (with $B=0$ and $A$ being random), the initial gradients will be massive and chaotic. A **Linear Warmup** (`warmup_ratio=0.1`) forces the learning rate to start at 0 and slowly climb to `3e-4` over the first 10% of training, preventing the model weights from exploding.

### Gradient Clipping
Even with warmup, code-mixed sequences can occasionally produce massive loss spikes. **Gradient Clipping** (`max_grad_norm=1.0`) physically caps the magnitude of the gradient vectors, ensuring no single bad training step permanently destroys the model.

### Early Stopping & Checkpoint Selection
We evaluate the model at the end of every epoch. The checkpoint strategy saves a new model folder for every epoch. `load_best_model_at_end=True` ensures that even if the model overfits and degrades in Epoch 3, the `GenerationTrainer` will automatically reload the best weights from Epoch 2 (based on `eval_loss`) before saving the final artifact.
