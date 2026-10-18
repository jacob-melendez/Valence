import logging

import numpy as np

from valence.governance.logfilter import install


def test_raw_media_records_are_dropped(caplog):
    logger = logging.getLogger("valence.test.logfilter")
    flt = install(logger)
    caplog.handler.addFilter(flt)
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("frame %s", np.zeros((2, 2)))
        logger.info("audio %r", b"\x00\x01")
        logger.info("nested %s", {"buf": [np.ones(3)]})
        logger.info("tick %d purged %d frames", 4, 50)
    assert [r.getMessage() for r in caplog.records] == ["tick 4 purged 50 frames"]
    assert flt.dropped >= 3


def test_install_filters_valence_child_loggers_without_root_handlers(capsys):
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        install()
        logging.getLogger("valence.some.module").warning("frame %s", np.zeros(3))
        logging.getLogger("valence.some.module").warning("tick done")
    finally:
        root.handlers[:] = saved
    err = capsys.readouterr().err
    assert "tick done" in err and "frame" not in err
