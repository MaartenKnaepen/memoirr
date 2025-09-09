from components.preprocessor.srt_preprocessor import SRTPreprocessor


def test_srt_preprocessor_component_runs_and_outputs_jsonl_and_stats():
    srt_text = (
        "1\n00:00:01,000 --> 00:00:02,000\n- Hello!\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nHi again\n"
    )
    comp = SRTPreprocessor()
    out = comp.run(srt_text=srt_text)
    lines = out["jsonl_lines"]
    stats = out["stats"]
    assert isinstance(lines, list) and len(lines) == 2
    assert isinstance(stats, dict) and stats.get("kept") == 2
