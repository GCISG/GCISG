import torch


def mean(res, key):
    return torch.stack([x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()


def mean_sum(res, key, count_key):
    total_sum = torch.stack([x[key] * x[count_key] for x in res]).sum(dim=0)

    return total_sum


def sum_on_key(res, key):
    total_sum = torch.stack([x[key] for x in res]).sum(dim=0)

    return total_sum


def accuracy_new(output, label, num_class):
    label = label.reshape([label.shape[0], 1])
    _, pred = output.topk(1, 1, True, True)
    gt = label
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]

    correct = pred.eq(gt)

    k = 1
    correct_k = correct[:k].float()
    per_class_correct_list = [
        correct_k[0][pred_class_idx[0]].sum() for pred_class_idx in pred_class_idx_list
    ]
    per_class_correct_array = torch.tensor(per_class_correct_list).to(label.device)
    gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
    gt_class_zeronumber_tensor = gt_class_number_tensor == 0
    gt_class_number_matrix = torch.tensor(gt_class_number_list).to(label.device).float()
    gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
    gt_class_acc[gt_class_zeronumber_tensor] = 0

    return gt_class_acc, gt_class_number_matrix
