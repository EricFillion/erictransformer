from erictransformer import EricChat
from tests.chat import CHAT_CALL_TEXT, CHAT_out_dir_MODEL, eric_chat, CHAT_CALL_ARGS
from tests.functions.save import t_save


def test_save():
    t_save(
        eric_model=eric_chat,
        eric_type=EricChat,
        call_input=CHAT_CALL_TEXT,
        save_path=CHAT_out_dir_MODEL,
        args = CHAT_CALL_ARGS
    )
