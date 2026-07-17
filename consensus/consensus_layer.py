import numpy as np
import torch


def apply_occlusion(data_np, occlusion_rate, seed=None):
    if occlusion_rate == 0.0:
        return data_np.copy()
    if seed is not None:
        np.random.seed(seed)
    data_out = data_np.copy()
    num_joints = data_out.shape[3]
    num_to_zero = max(1, int(round(num_joints * occlusion_rate)))
    joints_to_zero = np.random.choice(num_joints, num_to_zero, replace=False)
    data_out[:, :, :, joints_to_zero, :] = 0.0
    return data_out


def get_model_logits(model, data_np, device, batch_size=64):
    model.eval()
    all_logits = []
    N = data_np.shape[0]
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.tensor(data_np[start:end], dtype=torch.float32).to(device)
            output = model(batch)
            all_logits.append(output.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def softmax(logits):
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)


def consensus_fusion(logits_st, logits_ctr):
    probs_st = softmax(logits_st)
    probs_ctr = softmax(logits_ctr)
    H_st = entropy(probs_st)
    H_ctr = entropy(probs_ctr)
    W_st = np.exp(-H_st)
    W_ctr = np.exp(-H_ctr)
    sum_W = W_st + W_ctr
    W_st = W_st / sum_W
    W_ctr = W_ctr / sum_W
    probs_fused = W_st[:, None] * probs_st + W_ctr[:, None] * probs_ctr
    return np.argmax(probs_fused, axis=1)


def consensus_fusion_with_confidence(logits_st, logits_ctr, confidence_threshold=0.6):
    probs_st = softmax(logits_st)
    probs_ctr = softmax(logits_ctr)
    H_st = entropy(probs_st)
    H_ctr = entropy(probs_ctr)
    W_st = np.exp(-H_st)
    W_ctr = np.exp(-H_ctr)
    sum_W = W_st + W_ctr
    W_st = W_st / sum_W
    W_ctr = W_ctr / sum_W
    probs_fused = W_st[:, None] * probs_st + W_ctr[:, None] * probs_ctr
    predictions = np.argmax(probs_fused, axis=1)
    confidence = np.max(probs_fused, axis=1)
    safe_signal = (confidence >= confidence_threshold).astype(int)
    return predictions, confidence, safe_signal