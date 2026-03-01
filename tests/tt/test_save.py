from erictransformer import EricTextToText
from tests.functions.save import t_save
from tests.tt import TT_CALL_TEXT, TT_out_dir_MODEL, eric_tt, TT_CALL_ARGS


def test_save():
    t_save(
        eric_model=eric_tt,
        eric_type=EricTextToText,
        call_input=TT_CALL_TEXT,
        save_path=TT_out_dir_MODEL,
        args=TT_CALL_ARGS
    )
