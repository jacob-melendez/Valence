from pathlib import Path

from valence.governance.persistence_scan import scan_repo, scan_source

REPO = Path(__file__).resolve().parents[1]

BAD = '''
import cv2, numpy as np, wave, pickle, soundfile as sf
cv2.imwrite("f.png", frame)
w = cv2.VideoWriter("o.mp4", 0, 30, (640, 480))
np.save("frame.npy", frame)
wave.open("a.wav", "wb")
sf.write("a.wav", block, 16000)
pickle.dump(frame, f)
open("x.bin", "wb")
open("x.txt", "w")
'''


def test_scanner_flags_every_banned_call():
    found = scan_source(BAD, "perception/bad.py", allow_text_writes=False)
    assert len(found) == 8, [str(v) for v in found]


def test_edge_may_write_text_jsonl_only():
    src = 'f = open("data/real/run.jsonl", "a", encoding="utf-8")\nopen("raw.bin", "wb")\n'
    found = scan_source(src, "edge/run.py", allow_text_writes=True)
    assert [v.line for v in found] == [2]


def test_documented_exception_is_allowed_but_bare_pragma_is_not():
    ok = 'json.dump(constants, f)  # persistence-ok: calibration scalars only\n'
    bare = 'json.dump(constants, f)  # persistence-ok:\n'
    assert scan_source(ok, "perception/calibration.py", False) == []
    assert len(scan_source(bare, "perception/calibration.py", False)) == 1


def test_reads_are_allowed():
    assert scan_source('open("models/x.task", "rb")\nopen("cfg.yaml")\n', "p.py", False) == []


def test_repository_has_no_persistence_in_raw_media_paths():
    violations = scan_repo(REPO)
    assert violations == [], "\n".join(map(str, violations))


def test_scanner_catches_aliases_low_level_writes_and_copies():
    src = (
        "from cv2 import imwrite as w\n"
        "from numpy import save\n"
        "import os, shutil\n"
        "w('x.png', frame)\n"
        "save('f.npy', frame)\n"
        "os.write(fd, frame.tobytes())\n"
        "f = os.fdopen(fd, 'wb')\n"
        "shutil.copy('a', 'b')\n"
    )
    found = scan_source(src, "perception/x.py", allow_text_writes=False)
    assert [v.line for v in found] == [4, 5, 6, 7, 8], [str(v) for v in found]
