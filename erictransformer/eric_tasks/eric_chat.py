import textwrap
import threading
import traceback
from typing import Iterator, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    default_data_collator,
)
from ericsearch import EricSearch


from erictransformer.eval_models import EvalModel
from erictransformer.args import EricTrainArgs, EricEvalArgs
from erictransformer.eric_tasks.args import (
    CHATCallArgs,
    CHATTokArgs,
)
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
    generate_gen_kwargs,
    get_pad_eos,
)
from erictransformer.eric_tasks.results import CHATResult
from erictransformer.eric_tasks.tok.tok_functions import (
    get_max_in_len,
    tokenize_chat_template,
)
from erictransformer.eric_transformer import EricTransformer, EricTransformerArgs
from erictransformer.loops import EvalResult
from erictransformer.utils import get_model_components
from erictransformer.validator import CHATValidator


class EricChat(EricTransformer):
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel, None] = "openai/gpt-oss-20b",
        *,
        trust_remote_code: bool = False,
        tokenizer: Union[str, PreTrainedTokenizerBase] = None,
        eric_search: Optional[EricSearch] = None,
    ):
        model_class = AutoModelForCausalLM

        eric_args = EricTransformerArgs(
            model_name=model_name,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer
        )

        super().__init__(eric_args)

        self.task_validator = CHATValidator(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            tokenizer=tokenizer,
            logger=self.logger,
        )

        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("The tokenizer must include a chat template")

        self._data_collator = default_data_collator

        self.logger.info("Using tokenizer's built-in chat template.")
        if model_name:
            self.config = self.model.config

        if self.model is not None:
            self.pad_token_id, self.eos_token_id = get_pad_eos(
                self.tokenizer, self.model
            )

        if (
            model_name is not None
        ):  # we don't need to initialize these if a model is not provided.
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

        if self.model:
            self._prep_model()

        self.to_stream_tokens = []

    def _get_call_thread_streamer(
        self, messages: list[dict], args: CHATCallArgs = CHATCallArgs()
    ):
        mapped_messages = map_chat_roles(
            messages=messages,
            model_name=self.eric_args.model_name,
            model_type=self.model_type,
        )
        if self.model_type != "gpt_oss":

            input_ids = self.tokenizer.apply_chat_template(
                mapped_messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                mapped_messages, add_generation_prompt=True, tokenize=False
            )

            prompt += "<|channel|>analysis<|message|>"
            self.to_stream_tokens.append(self.text_streamer_handler.step("<|channel|>"))
            self.to_stream_tokens.append(self.text_streamer_handler.step("analysis"))
            self.to_stream_tokens.append(self.text_streamer_handler.step("<|message|>"))
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.model.device)

        attention_mask = torch.ones_like(
            input_ids, dtype=torch.long, device=self.model.device
        )

        gen_streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )

        gen_kwargs = generate_gen_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=gen_streamer,
            args=args,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        def thread_fn():
            try:
                self.model.generate(**gen_kwargs)
            except BaseException:
                # Bare except here is OK because we are in background thread
                # and therefore don't need to worry about
                # muting BaseExceptions such as KeyboardInterrupt

                err_str = traceback.format_exc()
                self.logger.error(f"Error in generate thread: {err_str}")
                gen_streamer.end()

        gen_thread = threading.Thread(target=thread_fn)
        return gen_thread, gen_streamer

    def __call__(
        self, text: Union[str, List[dict]], args: CHATCallArgs = CHATCallArgs()
    ) -> CHATResult:
        messages = format_messages(text)

        if self.eric_search is not None:
            search_query = create_search_prompt_chat(text)
            data_result = self.eric_search(search_query, args=args.search_args)

            if data_result:
                top_result = data_result[0]
                rag_content = formate_rag_content(text=text, data_result=top_result)
                rag_message = formate_rag_message(rag_content=rag_content)
                messages.insert(-1, rag_message)

        self._get_model_ready_inference()
        gen_thread, gen_streamer = self._get_call_thread_streamer(messages, args)
        gen_thread.start()
        out_text = []
        try:
            for text in gen_streamer:
                tokens = self.tokenizer.encode(text)
                for token in tokens:
                    token_string = self.tokenizer.decode(token)
                    stream_result = self.text_streamer_handler.step(token_string)
                    if stream_result:
                        if stream_result.marker == "text":
                            out_text.append(stream_result.text)
        finally:
            gen_thread.join()

        final_text = "".join(out_text)

        return CHATResult(text=final_text)

    def stream(
        self, text: Union[str, List[dict]], args: CHATCallArgs = CHATCallArgs()
    ) -> Iterator[CHATStreamResult]:
        messages = format_messages(text)

        if self.eric_search is not None:
            search_query = create_search_prompt_chat(text)
            yield CHATStreamResult(
                text="", marker="search", payload={"query": search_query}
            )

            data_result = self.eric_search(search_query, args=args.search_args)

            if data_result:
                top_result = data_result[0]
                yield CHATStreamResult(
                    text="",
                    marker="search_result",
                    payload={
                        "text": top_result.text,
                        "best_sentence": top_result.best_sentence,
                        "metadata": top_result.metadata,
                    },
                )

                rag_content = formate_rag_content(text=text, data_result=top_result)
                rag_message = formate_rag_message(rag_content=rag_content)
                messages.insert(-1, rag_message)

        self._get_model_ready_inference()

        gen_thread, gen_streamer = self._get_call_thread_streamer(messages, args)

        while self.to_stream_tokens:
            stream_result = self.to_stream_tokens.pop(0)
            if stream_result:
                yield stream_result

        gen_thread.start()
        try:
            for text in gen_streamer:
                tokens = self.tokenizer.encode(text)
                for token in tokens:
                    token_string = self.tokenizer.decode(token)
                    stream_result = self.text_streamer_handler.step(token_string)
                    if stream_result:
                        yield stream_result

        finally:
            gen_thread.join()

    def _tok_function(
        self,
        raw_dataset,
        args: CHATTokArgs = CHATTokArgs(),
        file_type: str = "jsonl",
        procs: Optional[int] = None,
    ) -> Dataset:
        max_in_len = get_max_in_len(args.max_len, self.tokenizer)

        return tokenize_chat_template(
            tokenizer=self.tokenizer,
            dataset=raw_dataset,
            max_len=max_in_len,
            bs=args.bs,
            procs=procs,
        )

    def train(
        self,
        train_path: str = "",
        args: EricTrainArgs = EricTrainArgs(),
        eval_path: str = "",
        *,
        resume_path: str = "",
    ):
        return super(EricChat, self).train(
            train_path, args, eval_path, resume_path=resume_path
        )

    def eval(
        self, eval_path: str = "", args: EricEvalArgs = EricEvalArgs()
    ) -> EvalResult:
        return super(EricChat, self).eval(eval_path=eval_path, args=args)

    def tok(self, path: str, out_dir: str, args: CHATTokArgs = CHATTokArgs()):
        return super(EricChat, self).tok(
            path=path, out_dir=out_dir, args=args
        )

    def _load_model_components(
        self,
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
        return get_model_components(
            model_name_path=self.eric_args.model_name,
            trust_remote_code=self.eric_args.trust_remote_code,
            model_class=self.eric_args.model_class,
            tokenizer_path=self.eric_args.tokenizer,
            precision=self.precision_type,
        )

    def _format_tokenized_example(self, example: dict) -> dict:
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }

    def _get_default_eval_models(self) -> List[EvalModel]:
        return []

    def _prep_model(self):
        args = CHATCallArgs()
        generation_config = GenerationConfig.from_model_config(self.model.config)
        generation_config.num_beams = 1
        generation_config.early_stopping = False
        generation_config.do_sample = True
        generation_config.min_len = args.min_len
        generation_config.max_len = args.max_len
        generation_config.temp = args.temp
        generation_config.top_p = args.top_p
        self.model.generation_config = generation_config

    def _get_readme(self, repo_id: str) -> str:
        readme_text = textwrap.dedent(f"""\
        ---
        tags:
        - erictransformer
        - eric-chat
        ---
        # {repo_id}
    
        ## Installation: 
        
        ```
        pip install erictransformer
        ```
        
        ## Usage 
        
        ```python
        from erictransformer import EricChat, CHATCallArgs
        
        eric_chat = EricChat(model_name="{repo_id}")

        text = 'Hello world'

        result = eric_chat(text)
        print(result.text)
        
        # Streaming is also possible (see docs)
        ```

        See Eric Transformer's [GitHub](https://github.com/ericfillion/erictransformer) for more information. 

        """)

        return readme_text
