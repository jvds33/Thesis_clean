import re
import numpy as np


def extract_spans_para(seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")

            combined_list = [index_ac, index_sp, index_at, index_ot]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 3:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start:combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot = result

            # if the aspect term is implicit
            if at.lower() == 'it':
                at = 'null'
        except ValueError:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}')
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, sp, ot))

    return quads


def compute_f1_scores(pred_pt, gold_pt, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    Uses exact matching with only lowercase conversion.
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    def normalize_text(text):
        """Normalize text for exact matching with only lowercase conversion"""
        if text is None:
            return "null"
        return str(text).lower().strip()

    def is_quad_match(pred_quad, gold_quad):
        """Check if a predicted quad matches a gold quad using exact matching with only lowercase conversion"""
        if len(pred_quad) != len(gold_quad):
            return False
            
        for idx, (pred_item, gold_item) in enumerate(zip(pred_quad, gold_quad)):
            pred_norm = normalize_text(pred_item)
            gold_norm = normalize_text(gold_item)
            
            # Exact match only
            if pred_norm != gold_norm:
                return False
                    
        # If we didn't return False, all components matched
        return True

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for p_quad in pred_pt[i]:
            for g_quad in gold_pt[i]:
                if is_quad_match(p_quad, g_quad):
                    n_tp += 1
                    break

    if verbose:
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(all_preds, all_labels, verbose=True):
    """
    Compute model performance
    Assumes all_preds and all_labels are lists of lists of processed quads.
    Each inner list corresponds to a sample, and contains tuples of (aspect, opinion, category, polarity).
    """
    # assert len(all_preds) == len(all_labels), (len(all_preds), len(all_labels))
    # num_samples = len(all_labels)

    # The inputs all_preds and all_labels are already processed lists of quads.
    # No need to call extract_spans_para here anymore.

    # if verbose:
    #     for i in range(min(num_samples, 10)):
    #         print("gold quads ", all_labels[i])
    #         print("pred quads ", all_preds[i])
    #         print()

    scores = compute_f1_scores(all_preds, all_labels, verbose=verbose)

    # Return scores. The original also returned all_labels, all_preds, but these are now the direct inputs.
    # For consistency with potential previous usage, we can return them, or just the scores.
    # Let's just return scores for simplicity, as the caller (infer.py) already has these lists.
    return scores # Original: return scores, all_labels, all_preds
