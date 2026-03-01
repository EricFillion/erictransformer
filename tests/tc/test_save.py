from erictransformer import EricTextClassification
from tests.functions.save import t_save
from tests.tc import TC_CALL_TEXT, TC_out_dir_MODEL, eric_tc


def test_save():
    t_save(
        eric_model=eric_tc,
        eric_type=EricTextClassification,
        call_input=TC_CALL_TEXT,
        save_path=TC_out_dir_MODEL,
        args=None
    )
