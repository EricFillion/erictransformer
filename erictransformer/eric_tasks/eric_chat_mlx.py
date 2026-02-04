import os
import tempfile
import textwrap
from typing import Iterator, List, Optional, Union

from huggingface_hub import HfApi
from transformers import AutoConfig
from ericsearch import EricSearch


try:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import save_model
except ImportError as err:
    raise ImportError("""
      Failed to import MLX. If you are using a Mac,
      try `pip install mlx-lm`. If you have a CUDA-compatible device,
      try `pip install mlx[cuda] mlx-lm`. Otherwise, try `pip install mlx[cpu] mlx-lm`
    """) from err

from erictransformer.exceptions import (
    EricNoModelError,
    EricPushError,
    EricSaveError,
)
from erictransformer.eric_tasks.args import CHATCallArgs
from erictransformer.eric_tasks.chat_stream_handlers import (
    CHATStreamResult,
    DefaultStreamHandler,
    GPTOSSSMHandler,
    SmolStreamHandler,
)
from erictransformer.eric_tasks.chat_templates import map_chat_roles
from erictransformer.eric_tasks.misc import (
    create_search_prompt_chat,
    format_messages,
    formate_rag_content,
    formate_rag_message,
)
from erictransformer.eric_tasks.results import CHATResult
from erictransformer.utils import et_get_logger


class EricChatMLX:
    def __init__(
        self,
        model_name: str = "EricFillion/smollm3-3b-mlx",
        *,
        eric_search: Optional[EricSearch] = None,
    ):
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        self.logger = et_get_logger()

        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The tokenizer must include a chat template")

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)

        if self.config.model_type == "smollm3":
            self.text_streamer_handler = SmolStreamHandler(self.tokenizer, self.logger)
            self.model_type = "smollm3"
        elif self.config.model_type == "gpt_oss":
            self.text_streamer_handler = GPTOSSSMHandler(self.tokenizer, self.logger)
            self.model_type = "gpt_oss"

        else:
            self.text_streamer_handler = DefaultStreamHandler(self.tokenizer)
            self.model_type = "default"

        if eric_search:
            self.eric_search = eric_search
        else:
            self.eric_search = None

        self.to_stream_tokens = []

    def _get_streamer_prompt(
        self, messages: Union[List[dict]], args: CHATCallArgs = CHATCallArgs()
    ):
        mapped_messages = map_chat_roles(
            messages=messages, model_name=self.model_name, model_type=self.model_type
        )

        prompt = self.tokenizer.apply_chat_template(
            mapped_messages, add_generation_prompt=True, tokenize=False
        )
        # always think.
        if self.model_type == "gpt_oss":
            prompt += "<|channel|>analysis<|message|>"
            self.to_stream_tokens.append(self.text_streamer_handler.step("<|channel|>"))
            self.to_stream_tokens.append(self.text_streamer_handler.step("analysis"))
            self.to_stream_tokens.append(self.text_streamer_handler.step("<|message|>"))


        sampler = make_sampler(
            temp=args.temp,
            top_p=args.top_p,
            top_k=args.top_k
        )

        return sampler, prompt

    def __call__(self, text:  Union[List[dict], str], args: CHATCallArgs = CHATCallArgs()) -> CHATResult:
        messages = format_messages(text)

        if self.eric_search is not None:
            search_query = create_search_prompt_chat(text)
            data_result = self.eric_search(search_query, args=args.search_args)

            if len(data_result):
                top_result = data_result[0]
                rag_content = formate_rag_content(
                    text=search_query, data_result=top_result
                )
                rag_message = formate_rag_message(rag_content=rag_content)
                messages.insert(-1, rag_message)

        sampler, prompt = self._get_streamer_prompt(messages=messages, args=args)
        out = []
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=args.max_len,
            sampler=sampler,
        ):
            stream_result = self.text_streamer_handler.step(resp.text)
            if stream_result:
                if stream_result.marker == "text":
                    out.append(resp.text)

        final_text = "".join(out).strip()
        return CHATResult(text=final_text)

    def stream(
        self, text: Union[List[dict], str], args: CHATCallArgs = CHATCallArgs()
    ) -> Iterator[CHATStreamResult]:
        messages = format_messages(text)

        if self.eric_search is not None:
            search_query = create_search_prompt_chat(text)
            yield CHATStreamResult(text=search_query, marker="search", payload={})
            data_result = self.eric_search(search_query, args=args.search_args)
            if data_result:
                top_result = data_result[0]

                yield CHATStreamResult(
                    text=top_result.text,
                    marker="search_result",
                    payload={
                        "best_sentence": top_result.best_sentence,
                        "metadata": top_result.metadata,
                    },
                )
                rag_content = formate_rag_content(
                    text=search_query, data_result=top_result
                )
                rag_message = formate_rag_message(rag_content=rag_content)
                messages.insert(-1, rag_message)

        sampler, prompt = self._get_streamer_prompt(messages=messages, args=args)

        while self.to_stream_tokens:
            stream_result = self.to_stream_tokens.pop(0)
            if stream_result:
                yield stream_result

        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=args.max_len,
            sampler=sampler,
        ):
            stream_result = self.text_streamer_handler.step(resp.text)
            if stream_result:
                yield stream_result

    def save(self, path: str):
        if self.model is None or self.tokenizer is None:
            raise EricNoModelError("No model/tokenizer loaded")

        os.makedirs(path, exist_ok=True)

        try:
            save_model(model=self.model, save_path=path)
        except Exception as e:
            raise EricSaveError(f"Error saving MLX model to {path}: {e}")

        try:
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise EricSaveError(f"error saving MLX tokenizer {path}: {e}")

        try:
            self.config.save_pretrained(path)
        except Exception as e:
            raise EricSaveError(f"Error saving config {path}: {e}")

    def push(self, repo_id: str, private: bool = True):
        api = HfApi()
        try:
            api.create_repo(repo_id, exist_ok=True, private=private)
        except Exception as e:
            self.logger.warning(f"Could not crate repo {e}")
            return
        try:
            has_readme = api.file_exists(repo_id, "README.md")
        except Exception as e:
            self.logger.warning(f"Could not info: {e}")
            return

        if not has_readme:
            readme_text = self._get_readme(repo_id)
            try:
                self.logger.info("Pushing README...")

                api.upload_file(
                    path_or_fileobj=readme_text.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )

            except Exception as e:
                # Don’t fail the whole push if README upload fails; just warn.
                self.logger.warning(f"Error pushing README: {e}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self.save(tmpdir)
                api.upload_large_folder(
                    repo_id, tmpdir, repo_type="model", private=True
                )

        except Exception as e:
            raise EricPushError(f"Error uploading model and tokenizer: {e}")

    def _get_readme(self, repo_id: str) -> str:
        readme_text = textwrap.dedent(f"""\
        ---
        tags:
        - erictransformer
        - eric-chat-mlx
        - mlx

        ---
        # {repo_id}

        ## Installation
        
        On Mac
        ```
        pip install mlx-lm erictransformer
        ```

        ## Usage 

        ```python
        from erictransformer import EricChatMLX, CHATCallArgs

        eric_chat = EricChatMLX(model_name="{repo_id}")
        
        text = 'Hello world'

        result = eric_chat(text)
        print(result.text)
        
        # Streaming is also possible (see docs)
        ```

        See Eric Transformer's [GitHub](https://github.com/ericfillion/erictransformer) for more information. 

        """)

        return readme_text
