from pycocotools.coco import COCO # Corrected import for COCO
from pycocoevalcap.eval import COCOEvalCap

def calculate_nlp_metrics(references_dict, hypotheses_dict, stage="val"):
    """
    Calculates NLP metrics (BLEU, METEOR, ROUGE_L, CIDEr) using pycocoevalcap.
    Args:
        references_dict (dict): A dictionary where keys are image_ids (UIDs) and
                                values are lists of reference strings.
                                e.g., {uid1: ['ref1 for uid1', 'ref2 for uid1'], uid2: ['ref1 for uid2']}
        hypotheses_dict (dict): A dictionary where keys are image_ids (UIDs) and
                                values are lists containing a single generated string.
                                e.g., {uid1: ['gen for uid1'], uid2: ['gen for uid2']}
        stage (str): Stage identifier (e.g., "val", "test") for COCO object creation.
                     This is to prevent issues if this function is called multiple times
                     with different datasets within the same COCO instance lifetime,
                     though for typical use (one call per validation/test) it might not be strictly necessary.
                     However, COCO class can have issues if not re-initialized properly.
    Returns:
        dict: A dictionary containing the calculated scores (e.g., 'BLEU_1', 'CIDEr').
    """
    if not references_dict or not hypotheses_dict:
        print("Warning: References or hypotheses are empty. Cannot calculate NLP metrics.")
        return {}

    # Check if UIDs match
    if set(references_dict.keys()) != set(hypotheses_dict.keys()):
        print("Warning: Mismatch in UIDs between references and hypotheses. Metrics will be partial or incorrect.")
        # Optionally, intersect keys or handle error more strictly

    shared_uids = list(set(references_dict.keys()) & set(hypotheses_dict.keys()))
    if not shared_uids:
        print("Warning: No common UIDs between references and hypotheses.")
        return {}

    # Format for COCO:
    # references_coco: list of dicts [{'image_id': uid, 'caption': ref_string, 'id': ann_id}]
    # hypotheses_coco: list of dicts [{'image_id': uid, 'caption': gen_string, 'id': ann_id}]
    # Note: 'id' here is annotation_id, must be unique for each caption.
    # 'image_id' is our UID, can be repeated if multiple references for one image.

    res_annotations = []
    hypo_annotations = []
    ann_id_counter = 0

    for uid in shared_uids:
        # For references
        for ref_cap in references_dict.get(uid, []):
            res_annotations.append({
                'image_id': uid,
                'caption': ref_cap,
                'id': ann_id_counter
            })
            ann_id_counter += 1

        # For hypotheses (COCO expects a list of captions, but we have one per image)
        # The hypothesis also needs an annotation ID, though it's less critical for its structure
        # as long as image_ids match up.
        # For COCOEvalCap, the hypothesis file is typically a list of {'image_id': uid, 'caption': gen_cap}
        # Let's ensure our hypotheses_dict values are lists of strings as expected.
        hypo_cap_list = hypotheses_dict.get(uid, [])
        if hypo_cap_list: # Should be one caption in the list
             hypo_annotations.append({
                'image_id': uid,
                'caption': hypo_cap_list[0] # Take the first (and only) hypothesis
                # 'id' field is not strictly needed for hypothesis file in COCOEvalCap from file,
                # but if building COCO object, it might be. Let's use image_id as a proxy for unique id here for hypo.
                # 'id': uid # This makes it unique per image, not per annotation. This is fine for hypo.
            })


    if not res_annotations or not hypo_annotations:
        print("Warning: No valid annotations formed for COCO. Skipping metrics.")
        return {}

    # Create COCO objects
    # Using a unique name for dataset if this function might be called multiple times
    # However, COCOEvalCap reinitializes its own COCO objects from these annotations if passed directly.

    # Create COCO object for references.
    coco_res = COCO()
    coco_res.dataset = {
        'info': {}, # Added empty info field
        'images': [{'id': uid} for uid in shared_uids],
        'annotations': res_annotations,
        'type': 'captions'
    }
    coco_res.createIndex()

    # Create COCO object for hypotheses.
    coco_hypo = COCO()
    coco_hypo.dataset = {
        'info': {}, # Added empty info field
        'images': [{'id': uid} for uid in shared_uids],
        'annotations': [],
        'type': 'captions'
    }
    coco_hypo.createIndex() # Create index for images

    # hypo_annotations is the list of results: [{'image_id': uid, 'caption': caption_string}]
    # loadRes expects this format. It will populate coco_hypo.dataset['annotations'] and re-index.
    if hypo_annotations: # Ensure it's not empty before calling loadRes
        coco_hypo.loadRes(hypo_annotations)
    # else: if hypo_annotations is empty, coco_hypo remains an empty COCO results object for these UIDs.
    # COCOEvalCap should handle this (e.g. by producing zero scores for metrics).

    coco_eval = COCOEvalCap(coco_res, coco_hypo) # Pass two COCO objects

    # params['image_id'] is already set to coco_res.getImgIds() by default in COCOEvalCap,
    # which should be our shared_uids if coco_res is built correctly.
    # So, explicitly setting coco_eval.params['image_id'] might not be necessary
    # unless we want to evaluate on a subset. For now, rely on default.

    # Evaluate
    try:
        coco_eval.evaluate()
    except Exception as e:
        print(f"Warning: Error during COCO full evaluation (possibly SPICE model download): {e}")
        print("SPICE scores might be missing. Other scores (BLEU, METEOR, ROUGE_L, CIDEr) might still be available.")
        # If coco_eval.eval is not populated at all or is missing critical keys,
        # we might need to return a partially filled dict or an error indicator.
        # For now, we'll proceed and return whatever is in coco_eval.eval.
        if not coco_eval.eval: # If eval dict is empty due to early crash
            print("Critical error: coco_eval.eval is empty after evaluate() call failed.")
            # Return an empty dict or specific error codes for metrics
            return {metric: 0.0 for metric in ["CIDEr", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "METEOR"]}


    # coco_eval.eval contains:
    # {
    #     'Bleu_1': score, 'Bleu_2': score, 'Bleu_3': score, 'Bleu_4': score,
    #     'METEOR': score, 'ROUGE_L': score, 'CIDEr': score, 'SPICE': score (if computed)
    # }
    return coco_eval.eval

if __name__ == '__main__':
    print("Testing NLP metrics calculation...")
    # Dummy data
    refs = {
        1: ["this is a test reference one for image one", "this is another reference for image one"],
        2: ["this is a reference for image two"],
        3: ["short ref three"]
    }
    hypos = {
        1: ["this is a generated test for image one"], # Matches one ref somewhat
        2: ["this is a generated for image two"],     # Matches somewhat
        3: ["completely different output"]            # No match
    }
    # Test with an extra hypothesis not in references
    hypos_extra = {**hypos, 4: ["this image was not in references"]}
    # Test with a missing hypothesis
    hypos_missing = {1: ["this is a generated test for image one"]}


    print("\n--- Test Case 1: Matching UIDs ---")
    scores = calculate_nlp_metrics(refs, hypos)
    if scores:
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
    else:
        print("Metrics calculation failed or returned empty.")

    print("\n--- Test Case 2: Extra hypothesis UID ---")
    scores_extra = calculate_nlp_metrics(refs, hypos_extra)
    if scores_extra:
        for metric, score in scores_extra.items():
            print(f"{metric}: {score:.4f}")
        assert set(scores_extra.keys()) == set(scores.keys()) if scores else True # Should produce same metrics for shared UIDs
    else:
        print("Metrics calculation failed or returned empty.")

    print("\n--- Test Case 3: Missing hypothesis UID ---")
    scores_missing = calculate_nlp_metrics(refs, hypos_missing)
    if scores_missing:
        for metric, score in scores_missing.items():
            print(f"{metric}: {score:.4f}")
    else:
        print("Metrics calculation failed or returned empty.")

    print("\n--- Test Case 4: Empty inputs ---")
    scores_empty_refs = calculate_nlp_metrics({}, hypos)
    assert not scores_empty_refs, "Should return empty dict for empty refs"
    print(f"Empty refs score: {scores_empty_refs}")
    scores_empty_hypos = calculate_nlp_metrics(refs, {})
    assert not scores_empty_hypos, "Should return empty dict for empty hypos"
    print(f"Empty hypos score: {scores_empty_hypos}")

    print("\nNLP metrics test completed.")
