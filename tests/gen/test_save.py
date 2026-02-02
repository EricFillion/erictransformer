from erictransformer import EricGeneration
from tests.functions.save import t_save
from tests.gen import GEN_CALL_TEXT, GEN_out_dir_MODEL, eric_gen, GEN_CALL_ARGS


def test_save():
    t_save(
        eric_model=eric_gen,
        eric_type=EricGeneration,
        call_input=GEN_CALL_TEXT,
        save_path=GEN_out_dir_MODEL,
        args=GEN_CALL_ARGS
    )
