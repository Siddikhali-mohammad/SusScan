import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.verdict import classify, Verdict

def test_screenshot_case():
    r = classify(0.59, 0.49, 0.26, 0.30)
    assert r.verdict == Verdict.UNCERTAIN, f"Expected UNCERTAIN, got {r.verdict} (rule={r.triggered_rule})"

def test_strong_real():
    r = classify(0.08, 0.12, 0.05, 0.08)
    assert r.verdict == Verdict.STRONG_REAL, f"Expected STRONG_REAL, got {r.verdict}"

def test_strong_fake():
    r = classify(0.89, 0.92, 0.85, 0.90)
    assert r.verdict == Verdict.STRONG_FAKE, f"Expected STRONG_FAKE, got {r.verdict}"

def test_likely_real():
    r = classify(0.20, 0.30, 0.42, 0.30)
    assert r.verdict == Verdict.LIKELY_REAL, f"Expected LIKELY_REAL, got {r.verdict}"

def test_likely_fake():
    r = classify(0.70, 0.60, 0.58, 0.65)
    assert r.verdict == Verdict.LIKELY_FAKE, f"Expected LIKELY_FAKE, got {r.verdict}"

def test_margin_abstain():
    r = classify(0.48, 0.52, 0.50, 0.50)
    assert r.verdict == Verdict.UNCERTAIN, f"Expected UNCERTAIN, got {r.verdict}"

def test_spread_abstain():
    # votes_fake=1 (only efb4), ensemble=0.50 → Rule 1 still fires
    r = classify(0.80, 0.50, 0.20, 0.50)
    assert r.verdict == Verdict.UNCERTAIN, f"Expected UNCERTAIN, got {r.verdict}"

def test_clip_dissents_strong_fake():
    # EFB4=99%, Swin=95%, CLIP=20%, ensemble=80% — real deepfake CLIP misses
    # Rule 1 exception: votes_fake=2 AND ensemble>0.65 → fall through to LIKELY_FAKE
    r = classify(0.99, 0.95, 0.20, 0.80)
    assert r.verdict == Verdict.LIKELY_FAKE, f"Expected LIKELY_FAKE, got {r.verdict} (rule={r.triggered_rule})"

def test_clip_dissents_likely_real():
    # EFB4=10%, Swin=6%, CLIP=55%, ensemble=14% — real image CLIP barely flags
    # Rule 1 exception: votes_real=2 AND ensemble<0.35 → fall through to LIKELY_REAL
    r = classify(0.10, 0.06, 0.55, 0.142)
    assert r.verdict == Verdict.LIKELY_REAL, f"Expected LIKELY_REAL, got {r.verdict} (rule={r.triggered_rule})"

if __name__ == "__main__":
    tests = [test_screenshot_case, test_strong_real, test_strong_fake,
             test_likely_real, test_likely_fake, test_margin_abstain, test_spread_abstain,
             test_clip_dissents_strong_fake, test_clip_dissents_likely_real]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
