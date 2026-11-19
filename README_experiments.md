# Experiments pack: controls, audit, held-out estimate, seeds, saliency

Seven scripts written against YOUR code (train.py, evaluate.py, dataset.py, protocol_utils.py, adversarial_attack.py).
Copy all of `scripts/` into your repo's `scripts/` folder (they import `adversarial_attack.py`, `src/attacks/protocol_utils.py`
and `src/encoding/flow_to_image.py` from the repo).

STATUS: only flow_audit.py (with audit_utils.py) was run, on synthetic data. The other five were syntax-checked, NOT run.
Run the smoke test in step 2 before the long runs.

| Script | Needs | Answers |
|---|---|---|
| flow_audit.py | flow cache only (no GPU, no model) | Why 282 of 600 flows produced no result; where the 0.36% invalid checksums come from |
| attack_controls.py | cache + model, GPU | Does the protocol constraint (not the attack) explain the modest success? Does optimisation beat random noise? |
| seen_unseen.py | cache + model + PCAPs | A held-out accuracy estimate for the existing model; explains the 0.975 vs 0.79 mismatch |
| train_seeds.py | PCAPs, GPU | Three more models with the same recipe (run-to-run variation) |
| aggregate_controls.py | controls_summary_*.csv | Mean +- std table and figure over models |
| saliency_analysis.py | cache + model, GPU | Where the classifier looks (header / payload / length / timing); does it predict who flips? |

## Steps (run from the project root, venv active)

1. Audit (seconds):
       python scripts/flow_audit.py
2. Smoke test of the GPU scripts (small and fast; delete the smoke outputs afterwards):
       python scripts/attack_controls.py --cap 20 --steps 5 --tag smoke
       del results\controls\*smoke*
   If anything errors, send me the traceback.
3. The main experiment (the model used for the report):
       python scripts/attack_controls.py
   Read the printed 'Flip rates' tables and the sanity check against the official run.
4. Held-out estimate and the saliency analysis:
       python scripts/seen_unseen.py --raw data/raw
       python scripts/saliency_analysis.py
5. Three more models, then the controls on each:
       python scripts/train_seeds.py --seeds 0 1 2
       python scripts/attack_controls.py --ckpt results/metrics/resnet18_seed0.pth --tag seed0
       python scripts/attack_controls.py --ckpt results/metrics/resnet18_seed1.pth --tag seed1
       python scripts/attack_controls.py --ckpt results/metrics/resnet18_seed2.pth --tag seed2
       python scripts/aggregate_controls.py
   (train_seeds.py builds the window dataset once with scapy, which is slow, and caches it in
   data/windows_cache.npz. That file is under data/, which git ignores.)
6. Commit:
       git add scripts results/audit results/controls results/saliency results/seen_unseen
       git status
       git commit -m "Add controls, audit, held-out estimate, seeds and saliency analysis"
       git push
   Check `git status` first: no .pth files and nothing from data/.

## How to read the controls (what each outcome means)
Compare flip rates on flows WITH editable payload (the comparable set), same epsilon:
- all_pixels >> payload: the protocol constraint is what limits the attack. This is the result the project is built on.
- payload ~ ch0_bytes ~ all_pixels: the constraint costs the attacker little here; the model is simply hard to flip at these budgets.
  Then say so plainly: realistic attacks are about as effective as unrealistic ones on this model.
- payload ~ random_payload: the optimisation adds almost nothing; the flips are noise sensitivity, not an optimised evasion.
- all_pixels flips the DoS-TCP flows (subset 'all'), payload cannot touch them: immunity is a property of the constraint.
Report whatever you find. Any of these is a valid, publishable finding if it is stated with its denominators.

## Corrections to apply to the Phase 3 report now (from reading your code)
Replace the [CONFIRM] markers with these:
- Training (Section 5): ResNet-18 from scratch (pretrained=False), 3 input channels, cross-entropy, Adam, learning rate 1e-3,
  batch size 32, 5 epochs, seed 42, deterministic cuDNN. All windows are used for training; there is NO validation or test split.
- Windows and support (Sections 4.4-4.6, 6): training windows are consecutive 16-packet chunks of a flow (last chunk may be
  partial), at most 20 windows per flow, flows with fewer than 4 packets dropped, only the first 50,000 IP packets of each PCAP
  file are read. 'Support' counts windows. evaluate.py re-extracts the same windows, so the baseline table IS a training-set fit.
- Two different samples (Section 6): the baseline is scored on windows from the first 50,000 packets; the attack uses a separate
  flow loader (up to 600 stored flows per class, scanned further into each file), one image per flow from its first 16 packets,
  and a sample of 150 per class. Some attacked flows therefore lie beyond the training prefix and were never seen. seen_unseen.py
  measures this. This replaces the three untested explanations in Section 6.
- Attack (Section 8.3): the reported predictions come from the RE-ENCODED flow (bytes written back, checksums repaired, flow
  re-encoded). For the unseen class the label is the model's own clean prediction (so 'prediction changed', not 'flip').
- Realization difference (Section 8.4): the optimizer moves in whole byte values, so a difference of 0.0000 mostly confirms that
  writing back and re-encoding is lossless. Say that; do not present it as an independent proof of realizability.
- Checksums (Section 8.4): verify_checksums checks the IP header checksum and the TCP/UDP checksum. repair_transport_checksum
  fixes only the transport checksum of packets whose bytes changed. So an invalid packet is one whose IP header checksum was
  already wrong, or an unedited packet whose transport checksum was already wrong in the capture (typical of checksum offload).
  The constant 0.9964 across all epsilons fits this. flow_audit.py tests it: if the pre-edit validity is about 0.9964 the
  failures are inherited from the capture.

## Known limits
- All of these evaluate in image space or on the stored flows; nothing here shows that the edited traffic still performs its attack.
- Seeds vary the trained model only; like train.py they train and report on the same data.
- seen_unseen.py compares later flows of the same captures (same attacker devices): an estimate, not a clean split.
- Class indices are read from the checkpoint, except where a script assumes 1 DDoS / 2 DoS-TCP / 3 Recon for the official
  per-sample file (stated in the output).
